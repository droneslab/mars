import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import symbol_shorthand
L = symbol_shorthand.L
X = symbol_shorthand.X
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

annofile = sys.argv[1]
train_ds = sys.argv[2]
fa_loss = sys.argv[3]
ma_loss = sys.argv[4]
ml_loss = sys.argv[5]

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
    model = MetricLDN.load_from_checkpoint(Path(path) / "model.ckpt", args=args, num_classes=0).eval()
    return model

def get_detections(line, transform):
    splits = line.split(' ')
    img_path = splits[0]
    boxes = splits[1:] # each box is in l,r,t,b format
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get all boxes and NMS
    parsed = np.zeros((len(boxes), 5))
    for i in range(len(boxes)):
        box = boxes[i]
        b_vals = box.split(',')
        idx = b_vals[0]
        # box vals are l,r,t,b
        l,r,t,b = [int(x) for x in b_vals[1:]]
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

def get_matching_pts(frame_info, matched_ids, frame_num, draw_matches=False, last_img=None, curr_img=None, kp_window_size=10):
    if draw_matches:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    
    pts1 = []
    pts2 = []
    for matched_id in matched_ids:
        last_box = [x for x in frame_info[frame_num-1]['boxes'] if x[0] == matched_id][0][1]
        curr_box = [x for x in frame_info[frame_num]['boxes'] if x[0] == matched_id][0][1]
        ll, lr, lt, lb = last_box
        cl, cr, ct, cb = curr_box
                
        lcenter_x = int(ll+((lr-ll)//2))
        lcenter_y = int(lt+((lb-lt)//2))
        ccenter_x = int(cl+((cr-cl)//2))
        ccenter_y = int(ct+((cb-ct)//2))
        
        if draw_matches:
            cv2.rectangle(last_img, (ll,lt), (lr,lb), (0,0,255), 1)
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
        ax1.imshow(last_img, cmap='gray', vmin=np.min(last_img), vmax=np.max(last_img))
        ax2.imshow(curr_img, cmap='gray', vmin=np.min(curr_img), vmax=np.max(curr_img))
        plt.show()
        plt.close()
    return pts1, pts2

def calc_pose(pts1, pts2):
    E,_ = cv2.findEssentialMat(np.array(pts1), np.array(pts2))
    inliers, R, t, mask = cv2.recoverPose(E, np.array(pts1), np.array(pts2))
    return R,t

# Brute-force loop closure check
def bf_loop_closure(frame_info, frame_num, percent_intersection=0.75):
    # For current frame and all frames at least 10 away, check for loop closure criteria
    # LOOP CLOSURE CRITERIA BETWEEN TWO FRAMES
    # 1.) Number of intersecting detections is >= threshold (number of current detections*percent_intersection)
    # 2.) Threshold is at least 5 detections
    curr_ids = set(frame_info[frame_num]['ids'])
    thresh = len(curr_ids)*percent_intersection
    if thresh < 5:
        return []
    candidate_frames = []
    for num in range(0, frame_num-10):
        test_ids = set(frame_info[num]['ids'])
        matches = curr_ids & test_ids
        if len(matches) >= thresh and num <= frame_num-10:
            candidate_frames.append((len(matches), num))
    candidate_frames.sort(key=lambda x: x[0])
    # Return lowest frame_number which satisfies criteria
    return candidate_frames[0] if candidate_frames else []

def report_on_progress(graph: gtsam.NonlinearFactorGraph, current_estimate: gtsam.Values,
                        key: int):
    """Print and plot incremental progress of the robot for 2D Pose SLAM using iSAM2."""

    # Print the current estimates computed using iSAM2.
    print("*"*50 + f"\nInference after State {key+1}:\n")
    print(current_estimate)

    # Compute the marginals for all states in the graph.
    marginals = gtsam.Marginals(graph, current_estimate)

    # Plot the newly updated iSAM2 inference.
    fig = plt.figure(0)
    if not fig.axes:
        axes = fig.add_subplot(projection='3d')
    else:
        axes = fig.axes[0]
    plt.cla()

    i = 1
    while current_estimate.exists(i):
        gtsam_plot.plot_pose3(0, current_estimate.atPose3(i), 10,
                                marginals.marginalCovariance(i))
        i += 1

    axes.set_xlim3d(-30, 45)
    axes.set_ylim3d(-30, 45)
    axes.set_zlim3d(-30, 45)
    plt.pause(1)

def main():
    model = get_model()
    transform = tf.Compose([tf.ToTensor(), tf.Resize((128,128)),])
    index = faiss.IndexFlatIP(512)
    y_index = torch.empty(0) # to keep track of ids
    frame_num = 0
    frame_info = {}
    last_img = None
    last_pose = None
    
    # Define the camera calibration parameters
    K = gtsam.Cal3_S2(39.6, 512, 512)
    # Define the camera observation (landmark) noise model
    landmark_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v
    # Define the pose (odometry) noise model
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])) # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    
    # Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
    prior_xyz_sigma = 0.3
    # Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
    prior_rpy_sigma = 5
    # Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
    odometry_xyz_sigma = 0.2
    # Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
    odometry_rpy_sigma = 5
    # Create noise models
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_rpy_sigma*np.pi/180,
                                                                prior_rpy_sigma*np.pi/180,
                                                                prior_rpy_sigma*np.pi/180,
                                                                prior_xyz_sigma,
                                                                prior_xyz_sigma,
                                                                prior_xyz_sigma]))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma]))
    
    # Create a Nonlinear factor graph as well as the data structure to hold state estimates.
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # Create iSAM2 parameters which can adjust the threshold necessary to force relinearization and how many
    # update calls are required to perform the relinearization.
    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(0.1)
    parameters.relinearizeSkip = 1
    isam = gtsam.ISAM2(parameters)
    
    # Add the prior factor to the factor graph, and poorly initialize the prior pose to demonstrate iSAM2 incremental optimization.
    prior_R = gtsam.Rot3(0,0,0,0,0,0,0,0,0)
    prior_t = gtsam.Point3(0,0,0)
    prior_pose = gtsam.Pose3(prior_R, prior_t)
    graph.push_back(gtsam.PriorFactorPose3(1, prior_pose, PRIOR_NOISE))
    initial_estimate.insert(1, prior_pose.compose(gtsam.Pose3(
        gtsam.Rot3.Rodrigues(0.0, 0.0, 0.0), gtsam.Point3(0.0, 0.0, 0.0))))
    
    # Initialize the current estimate which is used during the incremental inference loop.
    current_estimate = initial_estimate
    
    with alive_bar(len(lines)) as bar:
        for line in anno_lines:
            frame_info[frame_num] = {}
            img, noisy_boxes, detections, truth_ids = get_detections(line, transform)
            
            # Index matching
            embeddings = F.normalize(model(detections))
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
            
            # Compute R,t on matches to last frame
            if frame_num != 0:
                curr_frame_ids = set(pred_ids)
                prev_frame_ids = set(frame_info[frame_num-1]['ids'])
                matched_ids = curr_frame_ids & prev_frame_ids
                
                if len(matched_ids) > 0:
                    # pts1, pts2 = get_matching_pts(frame_info, matched_ids, frame_num, draw_matches=True, last_img=last_img, curr_img=img)
                    pts1, pts2 = get_matching_pts(frame_info, matched_ids, frame_num)
                    R,t = calc_pose(pts1, pts2)
                    R = gtsam.Rot3(R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2])
                    t = gtsam.Point3(t.squeeze())
                    pose = gtsam.Pose3(R,t)
                else:
                    pose = last_pose
            else:
                R = gtsam.Rot3(0,0,0,0,0,0,0,0,0)
                t = gtsam.Point3(0,0,0)
                pose = gtsam.Pose3(R,t)
                    
            # Check for loop closure
            loop_closure_candidate = bf_loop_closure(frame_info, frame_num)

            # Add a binary factor in between two existing states if loop closure is detected.
            # Otherwise, add a binary factor between a newly observed state and the previous state.
            if loop_closure_candidate:
                loop_num = loop_closure_candidate[1]
                graph.push_back(gtsam.BetweenFactorPose3(frame_num+1, loop_num+1, pose, ODOMETRY_NOISE))
            else:
                graph.push_back(gtsam.BetweenFactorPose3(frame_num+1, frame_num+2, pose, ODOMETRY_NOISE))
                # Compute and insert the initialization estimate for the current pose using a noisy odometry measurement.
                noisy_estimate = current_estimate.atPose3(frame_num+1).compose(pose)
                initial_estimate.insert(frame_num+2, noisy_estimate)
                
            # Perform incremental update to iSAM2's internal Bayes tree, optimizing only the affected variables.
            isam.update(graph, initial_estimate)
            current_estimate = isam.calculateEstimate()
            # Report all current state estimates from the iSAM2 optimization.
            # report_on_progress(graph, current_estimate, frame_num)
            initial_estimate.clear()
            
            last_pose = pose
            last_img = img
            frame_num += 1
            bar()
            
     # Print the final covariance matrix for each pose after completing inference.
    marginals = gtsam.Marginals(graph, current_estimate)
    i = 1
    while current_estimate.exists(i):
        print(f"X{i} covariance:\n{marginals.marginalCovariance(i)}\n")
        i += 1

    plt.ioff()
    plt.show()



if __name__ == '__main__':
    main()