# Resume to Job Description Matching: Siamese BiLSTM

A Keras notebook that builds a Siamese (twin-tower) neural network to score how well a resume matches a job description on three levels: 0 (weak), 1 (medium), 2 (strong).

Built as a course project at E&ICT Academy, IIT Roorkee (Machine Learning & Agentic AI program), with AI coding assistance.

## Status

The architecture and training pipeline run end to end, but **the notebook currently uses synthetic placeholder data** (1,000 rows of repeated sentences with random labels), so the model has not learned anything meaningful (accuracy is about 33%, which is chance for three classes). Training on a real resume and job description dataset is the next step. To do that, replace the placeholder block in the data-loading cell with a CSV that has the columns `resume_text`, `jd_text` and `match_label`.

## Model design

- **Preprocessing:** regex text cleaning, a tokenizer limited to 15,000 words, and sequences padded to 140 tokens.
- **Twin towers (shared weights):** Embedding with masking, then a Bidirectional LSTM, then global max pooling. Each text becomes one vector: `u` for the resume and `v` for the job description.
- **Comparison features:** `|u - v|` (absolute difference) and `u * v` (element-wise product), concatenated into one vector.
- **Classifier head:** dense layers with Dropout (0.3) and a 3-class softmax output.
- **Training setup:** stratified 70/15/15 split, class weights, Adam optimizer, sparse categorical cross-entropy, and early stopping on validation loss.
- The custom layers are registered as serializable so the model can be saved and reloaded.

## Run it

```bash
pip install tensorflow pandas numpy scikit-learn jupyter
jupyter notebook Aamir_Quraishi_Day19_Assignment.ipynb
```

## Files

- `Aamir_Quraishi_Day19_Assignment.ipynb`: model and training pipeline
- `Deep Neural Network Resume.docx`: supporting document
