import faiss
import torch
import numpy as np
import torch.nn.functional as F


class FaissMatcher():
    def __init__(self, match_thresh, gpu=True):
        self.gpu = gpu
        self.match_thresh = match_thresh
        self.init()
        
    def init(self):
        # Vars for statistics tracking
        self.correct_matches = 0
        self.incorrect_matches = 0
        self.y_index = torch.empty(0)
        self.all_dists = []
        self.frame_num = 0
        
        # Index setup
        self.index = faiss.IndexFlatIP(512)
        if self.gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            self.y_index = self.y_index.cuda()
        self.index.reset()
        
    def reset(self):
        self.init()
    
    # Add to index
    def add(self, embs, truth_ids):
        embs = embs.cuda() if self.gpu else embs.cpu()
        truth_ids = truth_ids.cuda() if self.gpu else truth_ids.cpu()
        embs = F.normalize(embs)
        self.index.add(embs)
        # Keep ongoing record of ids we add to the database (for missed matches calculation)
        self.y_index = torch.cat((self.y_index, truth_ids), 0)
        
    # obtain TRUE matches to index
    def match(self, embs, truth_ids):
        embs = embs.cuda() if self.gpu else embs.cpu()
        truth_ids = truth_ids.cuda() if self.gpu else truth_ids.cpu()
        embs = F.normalize(embs)
        dists, ids = self.index.search(embs, 1)
        match_checks = torch.where(dists < self.match_thresh, -1, ids) # Cosine similarity
        matched_ids_pred  = self.y_index[ids[(match_checks != -1).nonzero()[:,0]].squeeze() ]
        matched_ids_truth = truth_ids[(match_checks != -1).nonzero()[:,0]]
        return matched_ids_pred, matched_ids_truth
        
    # Given a set of embeddings and truth labels, will match/add and compute statistics
    def update(self, embs, truth_ids):
        embs = embs.cuda() if self.gpu else embs.cpu()
        truth_ids = truth_ids.cuda() if self.gpu else truth_ids.cpu()
        
        # Normalize and match embeddings
        embs = F.normalize(embs)
        dists, ids = self.index.search(embs, 1)
        
        # For embeddings not matched (ids == -1 OR dists <= match_threshold), add it to the database
        match_checks = torch.where(dists < self.match_thresh, -1, ids) # Cosine similarity
        non_matched_embs = embs[(match_checks == -1).nonzero()[:,0]]
        non_matched_ids = truth_ids[(match_checks == -1).nonzero()[:,0]]
        self.index.add(non_matched_embs)
        
        ''' Compute Statistics '''
        # Store all distances for statistics
        if self.frame_num != 0: # First frame dists will be extremely high as index is empty
            self.all_dists += list(dists.cpu().numpy())
        
        # Keep ongoing record of ids we add to the database (for missed matches calculation)
        self.y_index = torch.cat((self.y_index, non_matched_ids), 0)
        
        # For embeddings matched (ids != -1), compute correct and incorrect matches        
        # index ids are sequential and y_index are truth. Use ids to index into y_index to get truth labels
        matched_ids_pred  = self.y_index[ids[(match_checks != -1).nonzero()[:,0]].squeeze() ]
        matched_ids_truth = truth_ids[(match_checks != -1).nonzero()[:,0]]
        self.correct_matches   += (matched_ids_pred == matched_ids_truth).sum()
        self.incorrect_matches += (matched_ids_pred != matched_ids_truth).sum()
        self.frame_num+=1
        
    def compute_results(self, view=True):
        # Duplicate feature vectors that are added to the index
        # i.e. ones we should have matched but didnt
        missed_matches = self.y_index.shape[0] - torch.unique(self.y_index).shape[0]
        total_matches = self.correct_matches+self.incorrect_matches
        self.results = {
            'feats_in_index': float(self.index.ntotal),
            'mean_cosine_similarity': float(np.mean(np.array(self.all_dists).flatten().squeeze())),
            'correct_matches': float(self.correct_matches.item()),
            'incorrect_matches': float(self.incorrect_matches.item()),
            'missed_matches': float(missed_matches),
            'index_matching_accuracy': (self.correct_matches/total_matches).item()*100,
            'total_recognition_accuracy': (self.correct_matches/(total_matches+missed_matches)).item()*100,
        }
        
        if view:
            print(f" ---------- IndexFlatIP (512), Matching threshold: {self.match_thresh} ----------")
            print("    Features in index:           {}".format(self.results['feats_in_index']))
            print("    Mean Cosine Similarity:      {:.4f}".format(self.results['mean_cosine_similarity']))
            print("    Correct matches:             {}".format(self.results['correct_matches']))
            print("    Incorrect matches:           {}".format(self.results['incorrect_matches']))
            print("    Missed matches:              {}".format(self.results['missed_matches']))
            print("    Index matching accuracy:     {:.4f}%".format(self.results['index_matching_accuracy']))
            print("    Total recognition accuracy:  {:.4f}%".format(self.results['total_recognition_accuracy']))