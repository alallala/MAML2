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
The file size is of 2GB so I suggest to load it on drive and access it from there, otherwise to change the location of the file go to task_generator.py in the __init__ method of the TaskGenerator class.
   
   
   
5. Run the main python script

   ```
   cd scripts/image_segmentation
   # For train 1-way 5-shot on clarity
   python main.py --dataset=clarity --mode=train --n_way=5 --k_shot=5 --k_query=5 --inner_lr=0.001 --meta_batchsz=2 --total_batches=50 --update_steps=1 --update_steps_test=1 --test_steps=10 --ckpt_steps=10 --print_steps=1

   # For test just use --mode=test 
   ```

      

## References

This project is the extension and adaptation of the original implementation: [HilbertXu/MAML-Tensorflow](https://github.com/HilbertXu/MAML-Tensorflow) in TensorFlow 2. 
The Unet model and the related architectures come from the original implementation: [qubvel/segmentation_models] (https://github.com/qubvel/segmentation_models).
