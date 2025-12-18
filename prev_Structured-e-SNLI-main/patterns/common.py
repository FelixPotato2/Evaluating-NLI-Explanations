from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
import ast
import nltk
import spacy
import string
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
# nltk.download('wordnet')
# nltk.download('stopwords')  # Download the stopwords dataset
# nltk.download('omw-1.4')
from typing import Dict, List, Tuple, Union
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
import pyinflect
nlp = spacy.load("en_core_web_sm")


