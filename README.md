# Real-time Trainee Action Recognition using Action Similarity Embeddings

### Few-shot action recognition framework

![framework](https://user-images.githubusercontent.com/25244764/209555208-bd1c26fb-c6bf-4e84-9bb2-c2c7d9b6c4bf.png)

### Overall pipeline

- Prepare body part embedding (BPE) model (https://github.com/chico2121/bpe)
- Extracts human skeletons from video frames using pre-trained pose estimator.
- To create **Standard Action DB**, compute action similarity embeddings for each video in training set using BPE model.
- Given test sample, recognize action following below steps:
1. Extract body part embeddings following same process when creating standard action DB.
2. Perform time alignment between test embedding and train embeddings using dynamic time warping (DTW) algorithm.
3. Calculate cosine distances between embeddings.
4. Predict actions using K-nearest neighbors (K-NN) algorithm.

### Results

#### Accuracy
<img src="https://user-images.githubusercontent.com/25244764/209556447-1d2332e2-7860-4ce5-9f0f-ff08d6fb93d0.png" width="30%" height="30%">

#### Throughput

<img src="https://user-images.githubusercontent.com/25244764/209556631-e88376b2-a492-4018-b760-48272244d115.png" width="30%" height="30%">
