import gtsam
from gtsam import symbol_shorthand
L = symbol_shorthand.L
X = symbol_shorthand.X
from gtsam.utils import plot
import os
import sys
sys.path.append(os.path.abspath('../src/'))
import wandb
wb_api = wandb.Api()
import argparse
from pathlib import Path
from model import MetricLDN
import torchvision.transforms as tf
from matcher import FaissMatcher
from alive_progress import alive_bar
import cv2
import numpy as np
from utils import non_max_suppression_fast
import torch
from eval import add_detection_noise
import torch.nn.functional as F
import faiss
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import math

annofile = sys.argv[1]
train_ds = sys.argv[2]
fa_loss = sys.argv[3]
ma_loss = sys.argv[4]
ml_loss = sys.argv[5]
num_classes = 350 if train_ds == 'hirise' else 0

with open(annofile, 'r') as f:
    lines = f.readlines()
anno_lines = [l.strip() for l in lines]

def get_model():
    run = wb_api.runs(
        "tjchase34/corl_23",
        {'config.dataset': train_ds,
        'config.epochs': 50,
        'config.args/ml_loss': ml_loss,
        'config.args/ma_loss': ma_loss,
        'config.args/fa_loss': fa_loss}
    )[0]
    model_path = f'model-{run.id}:v0'
    wb = wandb.init(project='corl_23', id=run.id, resume='must')
    path = wandb.use_artifact(f'tjchase34/corl_23/{model_path}').download()
    args = argparse.Namespace(attention='se', fa_loss=fa_loss, ma_loss='none', ml_loss=ml_loss, caml_theta=0.99)
    model = MetricLDN.load_from_checkpoint(Path(path) / "model.ckpt", args=args, num_classes=num_classes).eval()
    return model

def get_detections(line, transform):
    splits = line.split(' ')
    img_path = splits[0]
    boxes = splits[1:] # each box is in l,r,t,b format
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get all boxes and NMS
    parsed = np.zeros((len(boxes), 5))
    # good_boxes = []
    for i in range(len(boxes)):
        box = boxes[i]
        b_vals = box.split(',')
        idx = b_vals[0]
        # box vals are l,r,t,b
        l,r,t,b = [int(x) for x in b_vals[1:]]
        # good_boxes.append((l,t,r,b,int(idx)))
        parsed[i,0] = l
        parsed[i,1] = t
        parsed[i,2] = r
        parsed[i,3] = b
        parsed[i,4] = idx
    good_boxes = non_max_suppression_fast(parsed, 0.5)
    # stack up a numpy array of box (detection) crops, gather ground truth crater num
    batch_detections = torch.empty((len(good_boxes),3,128,128))
    truth_ids = torch.empty((len(good_boxes)), dtype=torch.int64)
    noisy_boxes = []
    for i in range(len(good_boxes)):
        box = good_boxes[i]
        bvals = box[:-1]
        truth_idx = box[-1]
        truth_ids[i] = truth_idx
        l,t,r,b = bvals
        l,r,t,b = add_detection_noise(l,r,t,b) # Add noise to bbox detection
        noisy_boxes.append((l,r,t,b))
        # Crop frame at detection
        crop = img[t:b, l:r]
        cropT = transform(crop)
        batch_detections[i,:] = cropT
    return img, noisy_boxes, batch_detections, truth_ids

def get_matching_pts(frame_info, matched_ids, prev_num, curr_num, draw_matches=False, prev_img=None, curr_img=None, kp_window_size=10):
    if draw_matches:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    
    pts1 = []
    pts2 = []
    for matched_id in matched_ids:
        prev_box = [x for x in frame_info[prev_num]['boxes'] if x[0] == matched_id][0][1]
        curr_box = [x for x in frame_info[curr_num]['boxes'] if x[0] == matched_id][0][1]
        ll, lr, lt, lb = prev_box
        cl, cr, ct, cb = curr_box
                
        lcenter_x = int(ll+((lr-ll)//2))
        lcenter_y = int(lt+((lb-lt)//2))
        ccenter_x = int(cl+((cr-cl)//2))
        ccenter_y = int(ct+((cb-ct)//2))
        
        if draw_matches:
            cv2.rectangle(prev_img, (ll,lt), (lr,lb), (0,0,255), 1)
            cv2.rectangle(curr_img, (cl,ct), (cr,cb), (0,0,255), 1)
            con = ConnectionPatch(xyA=(ccenter_x, ccenter_y), xyB=(lcenter_x, lcenter_y), coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="lime")
            ax2.add_artist(con)
        
        # Center crops of kp_window_size will serve as keypoints for computing R,t
        # This is a flow-style way of matching
        lxs = np.arange(int(lcenter_x-kp_window_size//2), int(lcenter_x+kp_window_size//2))
        lys = np.arange(int(lcenter_y-kp_window_size//2), int(lcenter_y+kp_window_size//2))
        cxs = np.arange(int(ccenter_x-kp_window_size//2), int(ccenter_x+kp_window_size//2))
        cys = np.arange(int(ccenter_y-kp_window_size//2), int(ccenter_y+kp_window_size//2))
        
        lx_grid,ly_grid = np.meshgrid(lxs,lys)
        l_grid = list(zip(list(lx_grid.flatten()), list(ly_grid.flatten())))
        pts1 += l_grid
        cx_grid,cy_grid = np.meshgrid(cxs,cys)
        c_grid = list(zip(list(cx_grid.flatten()), list(cy_grid.flatten())))
        pts2 += c_grid
        
    if draw_matches:
        ax1.imshow(prev_img, cmap='gray', vmin=np.min(prev_img), vmax=np.max(prev_img))
        ax2.imshow(curr_img, cmap='gray', vmin=np.min(curr_img), vmax=np.max(curr_img))
        plt.show()
        plt.close()
    return pts1, pts2

def outside_spin(frame_num):
    if (frame_num < 200 or frame_num > 250) and (frame_num < 550 or frame_num > 600):
        return True
    else:
        return False

def filter_pose(R,t):
    # If dominant motion is in Z direction, bad
    x,y,z = t
    if abs(z) > abs(x) or abs(z) > abs(y):
        return False
    return True

# Calculate camera pose between two frames 
def calc_pose(frame_info, prev_num, curr_num, prev_img=None, curr_img=None):
    prev_pred_ids = frame_info[prev_num]['ids']
    curr_pred_ids = frame_info[curr_num]['ids']
    matched_ids = set(prev_pred_ids) & set(curr_pred_ids)
    # if len(matched_ids) > 0 and outside_spin(curr_num):
    if len(matched_ids) > 0:
        # pts1, pts2 = get_matching_pts(frame_info, matched_ids, prev_num, curr_num, draw_matches=True, prev_img=prev_img, curr_img=curr_img)
        pts1, pts2 = get_matching_pts(frame_info, matched_ids, prev_num, curr_num)
        E,_ = cv2.findEssentialMat(np.array(pts1), np.array(pts2))
        inliers, R, t, mask = cv2.recoverPose(E, np.array(pts1), np.array(pts2))
        R = gtsam.Rot3(R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2])
        t = gtsam.Point3(t.squeeze())
        if curr_num > 550 and curr_num < 600:
            print(t)
        return gtsam.Pose3(R,t) if filter_pose(R,t) else gtsam.Pose3()
    else:
        return gtsam.Pose3()

# def calc_pose(pts1, pts2):
#     
#     return R,t

# Brute-force loop closure check
def bf_loop_closure(frame_info, frame_num):
    # For current frame and all frames at least 10 away, check for loop closure
    curr_ids = set(frame_info[frame_num]['ids'])
    candidate_frames = []
    for num in range(frame_num-2):
        test_ids = set(frame_info[num]['ids'])
        thresh = max(len(curr_ids), len(test_ids))*0.5
        matches = curr_ids & test_ids
        if len(matches) >= thresh and thresh >= 5:
            candidate_frames.append((len(matches), num))
    candidate_frames.sort(key=lambda x: x[0])
    # Return lowest frame_number which satisfies criteria
    return candidate_frames[0] if candidate_frames else []
    
def get_box_centers(boxes):
    # boxes in l,r,t,b format
    centers = []
    for box in boxes:
        l,r,t,b = box
        x = l+((r-l)/2)
        y = t+((b-t)/2)
        centers.append((x,y))
    return centers

def get_gt_poses(pose_lines):
    gt_poses = []
    for line in pose_lines:
        vals = line.split(' ')
        kitti = [float(v) for v in vals]
        v = kitti
        t = gtsam.Point3(np.array([v[3],vals[7],vals[11]]))
        R = gtsam.Rot3(v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10])
        gt_poses.append(gtsam.Pose3(R,t))
    return gt_poses

def rot3_from_mat(m):
    return gtsam.Rot3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2])

def main():
    model = get_model().cuda()
    transform = tf.Compose([tf.ToTensor(), tf.Resize((128,128)),])
    index = faiss.IndexFlatIP(512)
    y_index = torch.empty(0) # to keep track of ids
    frame_info = {}
    prev_frame = None
    traj_t = None
    traj_R = None
    
    with open('../data/pose_csv/kitti_LROCWAC_22Sep06_000000_6H_10S_poses.csv', 'r') as f:
        gt_pose_lines = f.readlines()
    gt_poses = get_gt_poses(gt_pose_lines)
    
    # Define the camera calibration parameters
    K = gtsam.Cal3_S2(39.6, 512, 512)
    # Define the camera observation (landmark) noise model
    landmark_noise = gtsam.noiseModel.Isotropic.Sigma(2, 10.0)  # ten pixels in u and v
    # Define the pose (odometry) noise model
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])) # 0.1 rad std on roll,pitch,yaw and 0.1m on x,y,z
    # Create a Nonlinear factor graph as well as the data structure to hold state estimates.
    graph = gtsam.NonlinearFactorGraph()
    estimate = gtsam.Values()
    
    with alive_bar(len(lines)) as bar:
        for frame_num in range(len(anno_lines)):
            line = anno_lines[frame_num]
            frame_info[frame_num] = {}            
            frame, noisy_boxes, detections, truth_ids = get_detections(line, transform)
            
            # Index matching
            embeddings = F.normalize(model(detections.cuda())).cpu()
            dists, ids = index.search(embeddings, 1)
                        
            # For embeddings not matched (ids == -1 OR dists <= match_threshold), add it to the database
            match_checks = torch.where(dists < 0.9, -1, ids) # Cosine similarity
            non_matched_idxs = (match_checks == -1).nonzero()[:,0]
            non_matched_embs = embeddings[non_matched_idxs]
            new_pred_ids = torch.arange(index.ntotal, index.ntotal+non_matched_embs.shape[0])
            index.add(non_matched_embs)
            y_index = torch.cat((y_index, new_pred_ids), 0)
            
            # Gather list of (pred) ids that were detected this frame
            pred_ids = ids.flatten().numpy()
            new_pred_ids = new_pred_ids.flatten().numpy()
            non_matched_idxs = non_matched_idxs.flatten().numpy()
            
            if len(new_pred_ids) != 0:
                pred_ids[non_matched_idxs] = new_pred_ids
            frame_info[frame_num]['ids'] = pred_ids
            frame_info[frame_num]['boxes'] = list(zip(pred_ids, noisy_boxes))
            
            # Add pose factors
            if frame_num == 0:
                gt_pose = gt_poses[frame_num]
                factor = gtsam.PriorFactorPose3(X(frame_num), gt_pose, pose_noise)
                graph.push_back(factor)
                estimate.insert(X(frame_num), gt_pose)
                traj_t = gt_pose.translation()
                traj_R = gt_pose.rotation().matrix()
            else:
                # Add a binary factor in between two existing states if loop closure is detected.
                loop_closure_candidate = bf_loop_closure(frame_info, frame_num)
                # loop_closure_candidate = three_frame_loop_closure(frame_info, frame_num)
                if loop_closure_candidate:
                    loop_num = loop_closure_candidate[1]
                    print(f"Loop --> {loop_num}")
                    # Calc pose between looped frame and current frame
                    pose = calc_pose(frame_info, loop_num, frame_num)
                    # Identity pose in between factor as we assume small camera movement
                    factor = gtsam.BetweenFactorPose3(X(loop_num), X(frame_num), gt_poses[frame_num], pose_noise)
                    
                # Otherwise, add a binary factor between a newly observed state and the previous state.
                else:
                    pose = calc_pose(frame_info, frame_num-1, frame_num)
                    factor = gtsam.BetweenFactorPose3(X(frame_num-1), X(frame_num), gt_poses[frame_num], pose_noise)
                    # pose = calc_pose(frame_info, frame_num-1, frame_num, prev_frame, frame)
                graph.push_back(factor)
                estimate.insert(X(frame_num), pose)
                
                # R = pose.rotation().matrix()
                # t = np.expand_dims(pose.translation(), -1)
                # traj_t = t+(traj_R@t)
                # traj_R = R@traj_R
                # estimate.insert(X(frame_num), gtsam.Pose3(rot3_from_mat(traj_R), traj_t))
            
            # Add factors for each landmark observation
            # landmark_centers = get_box_centers(noisy_boxes) # Same length as pred_ids
            # for landmark_center,pred_id in list(zip(landmark_centers, pred_ids)):
            #     factor = gtsam.GenericProjectionFactorCal3_S2(landmark_center, landmark_noise, X(frame_num), L(pred_id), K)
            #     graph.push_back(factor)
                # Triangulate in 3D world space and add as value
            
            # NOTE: 500 looks decent w/ just Z filtering. 600 starts to bend backwards
            # if frame_num == 900:
            if frame_num == 600:
                break
            
            prev_frame = frame
            bar()
            
    graph.saveGraph('test', estimate)
    
    params = gtsam.GaussNewtonParams()
    params.setVerbosity('TERMINATION')
    params.setRelativeErrorTol(1e+20)
    params.setAbsoluteErrorTol(1e+20)
    params.setMaxIterations(100)
    optimizer = gtsam.GaussNewtonOptimizer(graph, estimate, params)
    
    # params = gtsam.DoglegParams()
    # params.setVerbosity('TERMINATION')
    # optimizer = gtsam.DoglegOptimizer(graph, estimate)
    
    # params = gtsam.LevenbergMarquardtParams()
    # params.setVerbosityLM("SUMMARY")
    # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, estimate)
    
    result = optimizer.optimize()
    # result.print('Final results:\n')
    print("initial error = ", graph.error(estimate))
    print("final error = ", graph.error(result))
    
    # kitti_eval_lines = ""
    # poses = gtsam.utilities.allPose3s(result)
    # for key in poses.keys():
    #     pose = poses.atPose3(key)
    #     R = pose.rotation().matrix()
    #     t = pose.translation()
    #     kitti_eval_lines += f"{R[0][0]} {R[0][1]} {R[0][2]} {t[0]} {R[1][0]} {R[1][1]} {R[1][2]} {t[1]} {R[2][0]} {R[2][1]} {R[2][2]} {t[2]}\n"
    # with open(f'{train_ds}-{fa_loss}-{ma_loss}-{ml_loss}_evalTraj.txt', 'w') as f:
    #     f.write(kitti_eval_lines)

    # marginals = gtsam.Marginals(graph, result)
    plot.plot_trajectory(1, result, scale=1)
    plot.set_axes_equal(1)
    plt.show()
            

if __name__ == '__main__':
    main()
    