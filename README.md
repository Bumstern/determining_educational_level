# Determining educational level
Pytorch code for my student research paper: "DETERMINING EDUCATION LEVEL BASED ON A PHOTO OF HUMAN FACE".

## Resources
1. Pytorch VGG-Face implementation - https://github.com/chi0tzp/PyVGGFace
2. VGG-Face model weights - https://www.robots.ox.ac.uk/~vgg/software/vgg_face/
3. Article - NOT PUBLISHED YET

## File usage
* convert_weights.py - converts LuaTorch weights and loads to VGG-Face model 
* training.py - model training logic
* create_embeds.py - creates embeddings of the model
* fine_tuning.py - trains fine-tuned model 
* SVM_training.py - fits SVM model on created VGG-Face embeddings
* cross-val.py - VGG-Face and SVM models cross-validation 
* visualize_data.py - visualisation of data distribution 
