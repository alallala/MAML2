# MAML-Tensorflow
Tensorflow r2.1 adaptation to semantic segmentation task of the Model-Agnostic Meta-Learning from this paper: 

[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)


## Project Requirements

1. python 3.x
2. Tensorflow r2.1
3. numpy 
4. matplotlib


## Dataset 

1. The dataset clarity is contained in a tiff file with a numpy array of shape (7260, 256, 256, 4). 
There are 7260 images with H = 256, W = 256 and C = 4 (B,G,R and binary mask).
   
   
   
5. Run the main python script

   ```
   cd scripts/image_segmentation
   # For train n-way y-shot on clarity
   python main.py --dataset=clarity --mode=train --n_way=n --k_shot=y --k_query=15
   # For test just use --mode=test 
   ```

      

## References

This project is the extension and adaptation of the original implementation: [HilbertXu/MAML-Tensorflow](https://github.com/HilbertXu/MAML-Tensorflow) in TensorFlow 2. 

