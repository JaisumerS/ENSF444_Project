
# ENSF444 Fake News Classification Project

## Interactive Interface – `ML_main.py`

The primary way to use our fake news detection system is through the `ML_main.py` script. This script provides a simple interface where users can input a news article's title and body text to determine whether it is **true** or **fake** news.

### How to Use:
1. Run the interface using the command:

```bash
python ML_main.py
```

2. Follow the on-screen prompts to enter a news title and text.
3. The system will process your input and return predictions from all three models.

### Optional: View Model Metrics in the Interface
If you're curious about how each model performs, the interface also gives you the option to display key evaluation metrics (like accuracy, precision, recall, and F1 score).  
**Note:** Generating these metrics may take a little time, especially depending on your system.

---

## Individual Model Notebooks

To see each of these models in more detail, we’ve provided three separate Jupyter files:

- `SVM_model.ipynb`
- `RandomForest_model.ipynb`
- `MultinomianNB_model.ipynb`

Each notebook contains:
- Data preprocessing steps  
- Model training and evaluation  
- Performance metrics and visualizations (e.g., accuracy, precision, recall, F1 score)

These notebooks are perfect for training models independently and analyzing their results in more detail.
