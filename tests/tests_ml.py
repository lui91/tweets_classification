from hypothesis.extra.pandas import data_frames, column
from hypothesis.extra.numpy import arrays
from hypothesis import given, settings
from hypothesis.strategies import composite, SearchStrategy, text
# from ..web_app.models import train_classifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Callable


@composite
def word_selector(draw: Callable[[SearchStrategy[str]], str], min_value: int = 5, max_value: int = 30):
    corpus = "I told you property-based testing is awesome :). Great video, Arjan! One approach I like a lot is to use a test oracle. When I have an optimized and therefore complicated implementation of an algorithm, I often also implement a simple version of the algorithm which is easy to understand, but not optimized and not efficient. I use Hypothesis to generate random input, run both implementations, and then assert that they return the same result."
    rand_words = draw(text(alphabet=corpus ,min_size=min_value, max_size=max_value))
    return rand_words

@given(word_selector())
def create_df(rand_words: list[str]):
    df = data_frames([column("A", elements=word_selector())])

# @given(word_selector())
# @settings(max_examples=3)
# def test_pipeline(test_data: np.array) -> None:
#     cols = [f"col_{x}" for x in range(test_data.shape[1])]
#     df = pd.DataFrame(data=test_data, columns=cols)
#     X = df.iloc[:, 0]
#     Y = df[df.columns[4:]]
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#     model = train_classifier.build_model()
#     model.fit(X_train, Y_train)
    
if __name__ == "__main__":
    print(create_df().example())