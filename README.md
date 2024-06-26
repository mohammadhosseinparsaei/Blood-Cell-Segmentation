<div class="container">
    <div class="title-container">
        <h1>Blood Cell Segmentation</h1>
    </div>
    <div class="image-container">
        <img src="https://github.com/mohammadhosseinparsaei/Blood-Cell-Segmentation/blob/main/sample%20image.png" alt="Blood Cell Sample" width="200" align="right">
    </div>
</div>

In this exercise, I tried to segment the blood cells present in microscopic images.
### Approach
For training and testing the image segmentation model, a U-Net neural network architecture has been employed using the Keras library.
### Data Access
The data is accessible and downloadable from [here](https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask).
### Citation: 
- **Title:** Automatic Segmentation of Blood Cells from Microscopic Slides: A Comparative Analysis
- **Authors:** Deponker Sarker Depto, Shazidur Rahman, Md. Mekayel Hosen, Mst Shapna Akter, Tamanna Rahman Reme, Aimon Rahman, Hasib Zunair, M. Sohel Rahman, M.R.C. Mahdy
- **Journal:** Tissue and Cell
- **Volume:** 73
- **Year:** 2021
- **Pages:** 101653
- **ISSN:** 0040-8166
- **DOI:** [10.1016/j.tice.2021.101653](https://doi.org/10.1016/j.tice.2021.101653)
- **Link:** [Read the article](https://www.sciencedirect.com/science/article/pii/S0040816621001695)

### Tools and Libraries Used
- Python 3.6.15
- numpy 1.19.2
- pandas 1.1.5
- opencv 3.4.2
- scikit-learn 0.24.2
- keras 2.3.1
- tqdm 4.64.1
- matplotlib 3.3.4

### Model Evaluation with Random Images
<img src="https://github.com/mohammadhosseinparsaei/Blood-Cell-Segmentation/blob/main/evaluation.png" alt="images" width="600"/>

### IoU & Dice coefficient Plot
<img src="https://github.com/mohammadhosseinparsaei/Blood-Cell-Segmentation/blob/main/iou_dice_plot.png" alt="IoU & Dice plot" width="600"/>
