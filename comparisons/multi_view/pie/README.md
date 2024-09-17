# PIE_pose_invariant_embedding
Implementation of the research paper [PIEs: Pose Invariant Embeddings (CVPR2019)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ho_PIEs_Pose_Invariant_Embeddings_CVPR_2019_paper.pdf) 
The result presented in the paper is averaged over 5 rounds, so it might be slightly different from the model provided.


## Usage

1. Install all required library
```
conda env create -f environment.yml --name <env_name>
```

2. Download ModelNet40 dataset 
```
download_modelnet40.sh
```

3. Download pretrained models from https://drive.google.com/drive/folders/1l9VASmcr2oRD0PKKgv222syhsVcpU290?usp=sharing and place the pretrain models according to the folder organization. 

4. Run cnn based methods (cnn, mvcnn, picnn) by: 
```
cd cnn_based
cd <cnn, mvcnn, picnn>
sh run.sh
```

5. Run proxy based methods (proxy, mvproxy, piproxy) by: 
```
cd proxy_based
cd <proxy, mvproxy, piproxy>
sh run.sh
```

6. Run triplet center based methods (triplet, mvtriplet, pitriplet) by: 
```
cd triplet_center_based
cd <triplet, mvtriplet, pitriplet>
sh run.sh
```

7. Check the result of pretrained models in log folder and log_robustness folder. Log_robustness folder shows the classification results from single view to all the views provided. Read run.sh for more information.


## Result for pretrained models
The proposed method is similar to single view based methods when only single view is given and similar to multiview based methods (mv___) when all views are given. The proposed method is more robusted to the number of view given during inference time.

### MVCNN
<table>
  <tr>
    <td colspan="2">Methods</td>
    <td colspan="2">Classification</td>
  </tr>
  <tr>
    <td colspan="2"></td>
    <td>Single view</td>
    <td>All views (12)</td>
  </tr>
  <tr>
    <td rowspan="3">CNN based</td> </td>
    <td>cnn</td>
	<td>84.66</td>
	<td>87.50</td>
  </tr>
   <tr>
    <td>mvcnn</td>
	<td>77.75</td>
	<td>89.75</td>
  </tr>
   <tr>
    <td>picnn</td>
	<td>85.70</td>
	<td>89.25</td>
  </tr>
  <tr>
    <td rowspan="3">Proxy based</td> </td>
    <td>proxy</td>
	<td>85.60</td>
	<td>88.62</td>
  </tr>
  <tr>
    <td>mvproxy</td>
	<td>78.39</td>
	<td>90.38</td>
  </tr>
  <tr>
    <td>piproxy</td>
	<td>85.49</td>
	<td>89.25</td>
  </tr>
  <tr>
    <td rowspan="3">Triplet center based</td> </td>
    <td>triplet</td>
	<td>85.23</td>
	<td>88.75</td>
  </tr>
  <tr>
    <td>mvtriplet</td>
	<td>76.94</td>
	<td>89.38</td>
  </tr>
  <tr>
    <td>pitriplet</td>
	<td>83.50</td>
	<td>89.38</td>
  </tr>
</table>


### ObjectPI

<table>
  <tr>
    <td colspan="2">Methods</td>
    <td colspan="2">Classification</td>
  </tr>
  <tr>
    <td colspan="2"></td>
    <td>Single view</td>
    <td>All views (12)</td>
  </tr>
  <tr>
    <td rowspan="3">CNN based</td> </td>
    <td>cnn</td>
	<td>65.82</td>
	<td>76.53</td>
  </tr>
   <tr>
    <td>mvcnn</td>
	<td>59.44</td>
	<td>77.55</td>
  </tr>
   <tr>
    <td>picnn</td>
	<td>67.60</td>
	<td>79.59</td>
  </tr>
  <tr>
    <td rowspan="3">Proxy based</td> </td>
    <td>proxy</td>
	<td>69.52</td>
	<td>79.59</td>
  </tr>
  <tr>
    <td>mvproxy</td>
	<td>64.03</td>
	<td>76.53</td>
  </tr>
  <tr>
    <td>piproxy</td>
	<td>68.62</td>
	<td>79.59</td>
  </tr>
  <tr>
    <td rowspan="3">Triplet center based</td> </td>
    <td>triplet</td>
	<td>70.79</td>
	<td>77.55</td>
  </tr>
  <tr>
    <td>mvtriplet</td>
	<td>63.65</td>
	<td>78.57</td>
  </tr>
  <tr>
    <td>pitriplet</td>
	<td>69.64</td>
	<td>75.51</td>
  </tr>
</table>

## Citation
If you mentioned the method in your research, please cite this article:
```
@InProceedings{Ho_2019_CVPR,
author = {Ho, Chih-Hui and Morgado, Pedro and Persekian, Amir and Vasconcelos, Nuno},
title = {PIEs: Pose Invariant Embeddings},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
