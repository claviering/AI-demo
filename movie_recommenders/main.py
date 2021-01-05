from typing import Dict, Text

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Ratings data.
ratings = tfds.load('movielens/100k-ratings', split="train")
# Features of all the available movies.
movies = tfds.load('movielens/100k-movies', split="train")

# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_id": tf.strings.to_number(x["movie_id"]),
    "user_id": tf.strings.to_number(x["user_id"])
})
movies = movies.map(lambda x: tf.strings.to_number(x["movie_id"]))

# Build a model.
class Model(tfrs.Model):

  def __init__(self):
    super().__init__()

    # Set up user representation.
    self.user_model = tf.keras.layers.Embedding(
        input_dim=2000, output_dim=64)
    # Set up movie representation.
    self.item_model = tf.keras.layers.Embedding(
        input_dim=2000, output_dim=64)
    # Set up a retrieval task and evaluation metrics over the
    # entire dataset of candidates.
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.item_model)
        )
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.item_model(features["movie_id"])

    return self.task(user_embeddings, movie_embeddings)


model = Model()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# Train.
model.fit(train.batch(4096), epochs=5)

# Evaluate.
model.evaluate(test.batch(4096), return_dict=True)