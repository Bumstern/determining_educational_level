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


## Work results
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2"></th>
    <th class="tg-c3ow" colspan="2">Facenet + SVM</th>
    <th class="tg-c3ow" colspan="2">VGG-Face + SVM</th>
    <th class="tg-baqh" rowspan="2">Fine-tuned <br>VGG-Face<br>30 epochs</th>
  </tr>
  <tr>
    <th class="tg-c3ow">SVM(C=1)</th>
    <th class="tg-c3ow">SVM(C=0.3)</th>
    <th class="tg-c3ow">SVM(C=1)</th>
    <th class="tg-c3ow">SVM(C=0.3)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Accuracy</td>
    <td class="tg-9wq8">&nbsp;&nbsp;&nbsp;<br>0.76&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-9wq8">&nbsp;&nbsp;&nbsp;<br>0.77&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-9wq8">&nbsp;&nbsp;&nbsp;<br>0.77&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-9wq8">&nbsp;&nbsp;&nbsp;<br>0.78&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">&nbsp;&nbsp;&nbsp;<br>0.74&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td class="tg-c3ow">F1</td>
    <td class="tg-9wq8">&nbsp;&nbsp;&nbsp;<br>0.78&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-9wq8">&nbsp;&nbsp;&nbsp;<br>0.81&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-9wq8">&nbsp;&nbsp;&nbsp;<br>0.78&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-9wq8">&nbsp;&nbsp;&nbsp;<br>0.79&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">&nbsp;&nbsp;&nbsp;<br>0.75&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td class="tg-baqh">AUC-ROC</td>
    <td class="tg-nrix">&nbsp;&nbsp;&nbsp;<br>0.85&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">&nbsp;&nbsp;&nbsp;<br>0.85&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">&nbsp;&nbsp;&nbsp;<br>0.83&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">&nbsp;&nbsp;&nbsp;<br>0.83&nbsp;&nbsp;&nbsp;</td>
    <td class="tg-nrix">&nbsp;&nbsp;&nbsp;<br>0.84&nbsp;&nbsp;&nbsp;</td>
  </tr>
</tbody>
</table>

| **PCA** | **T-SNE** |
| ------- | --------- |
| <img src="https://user-images.githubusercontent.com/36205247/167295900-846b4da8-cd3f-4375-b854-33d5606792b2.png" width="350"> | <img src="https://user-images.githubusercontent.com/36205247/167295905-938d8356-3b60-498c-99b6-73b22189eff3.png" width="350">


## Addtional information
Need to store dataset outside of project with folder name - "dataset". Dataset consists of two classes "With degree" and "Without degree".
Each class stores face images with aspect ratio 224x224. 
