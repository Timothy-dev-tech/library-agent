"""Intent classification, catalog entity extraction, and library answers."""
import re
from datetime import timedelta

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from . import db, nlp
from .config import CHATBOT_MIN_CONFIDENCE, FORECAST_HORIZON_WEEKS

INTENT_EXAMPLES = {
    "availability": [
        "is this book available", "do you have this title", "check if the book is in stock",
        "can i borrow this book", "how many copies are available", "is clean code available",
        "does the library have this book", "is there a copy of this title",
        "check availability for this book", "can i check out this title",
    ],
    "due_soon": [
        "what books are due this week", "which loans are due soon", "what do i need to return",
        "show books due in the next seven days", "what is due this week",
        "list my upcoming due dates", "which books should i return this week",
        "show loans due soon", "what is due next week", "when should i return my books",
    ],
    "overdue": [
        "what books are overdue", "show overdue loans", "which books have not been returned on time",
        "list overdue books", "do i have any overdue books", "show me late returns",
        "which loans are past their due date", "find overdue items", "my overdue loans",
        "who has overdue books",
    ],
    "recommendations": [
        "recommend a book for me", "what should i read next", "suggest books based on my history",
        "find something i might like", "recommend books in my interests",
        "what books are popular with readers like me", "give me reading suggestions",
        "show my personalized recommendations", "books similar to what i borrowed",
        "what would you recommend",
    ],
    "search": [
        "search for books about machine learning", "find a book on history",
        "something about machine learning for beginners", "look up titles about biology",
        "find books matching this topic", "search the catalog for algorithms",
        "i need a book about economics", "show me books on artificial intelligence",
        "find titles related to this subject", "look for a book about networking",
    ],
    "author_search": [
        "which books are available from this author", "show me books by this author",
        "what titles has this author written", "find available books by an author",
        "which titles can i borrow from this writer", "list books written by an author",
        "does the library have books by this author", "show available titles from a writer",
        "what books can i borrow by this author", "find all titles by an author",
    ],
    "forecast": [
        "forecast book demand", "predict which books will be borrowed next month",
        "which titles need more copies", "show the borrowing demand forecast",
        "predict demand by genre", "what books will be popular next week",
        "estimate future library borrowing", "show projected loans",
        "which category will have the most demand", "forecast borrowing for the next month",
    ],
    "help": [
        "help", "what can you do", "how do i use this assistant",
        "show available commands", "what can i ask you", "tell me how this works",
        "what features are available", "can you explain your services",
    ],
    "unknown": [
        "tell me a joke", "what is the weather outside", "write an essay for me",
        "translate this sentence", "what is the capital of france", "solve my math homework",
        "play a song", "tell me a story", "who won the game", "what time is it",
    ],
}

BOOK_TITLE_PATTERN = re.compile(r"""["']([^"']{2,})["']""")
STUDENT_ID_PATTERN = re.compile(r"\bstudent(?:\s*(?:id|number))?\s*#?\s*(\d+)\b", re.I)
NAME_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)*")
AUTHOR_QUERY_CUES = frozenset({
    "book", "books", "title", "titles", "available", "availability", "borrow",
    "borrowable", "stock", "written", "wrote", "works",
})


class LibraryChatbot:
    """Classify a library question and dispatch it to tested service modules."""

    def __init__(self, books: pd.DataFrame, search_engine, recommender, forecaster):
        self.books = books.reset_index(drop=True)
        self.search_engine = search_engine
        self.recommender = recommender
        self.forecaster = forecaster
        texts, labels = [], []
        for intent, examples in INTENT_EXAMPLES.items():
            texts.extend(examples)
            labels.extend([intent] * len(examples))
        self.classifier = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0, C=5),
        )
        self.classifier.fit(texts, labels)

    def classify(self, query: str) -> tuple[str, float]:
        """Return the most likely intent and its model probability."""
        entities = self.extract_entities(query)
        if entities["author"] and AUTHOR_QUERY_CUES.intersection(nlp.raw_words(query)):
            return "author_search", 1.0
        title = entities["book_title"]
        if title:
            query = re.sub(re.escape(title), "book", query, flags=re.IGNORECASE)
        probabilities = self.classifier.predict_proba([query])[0]
        position = int(probabilities.argmax())
        return str(self.classifier.classes_[position]), float(probabilities[position])

    def extract_entities(self, query: str) -> dict:
        """Extract a catalog title, author, and optional student identifier."""
        titles = self.books["title"].tolist()
        normalized_query = nlp.normalize(query)
        quoted = BOOK_TITLE_PATTERN.search(query)
        matched_title = None

        if quoted:
            quoted_normalized = nlp.normalize(quoted.group(1))
            for title in titles:
                if nlp.normalize(title) == quoted_normalized:
                    matched_title = title
                    break

        if matched_title is None:
            matches = [
                title for title in titles
                if nlp.normalize(title) and nlp.normalize(title) in normalized_query
            ]
            if matches:
                matched_title = max(matches, key=lambda title: len(nlp.normalize(title)))

        query_name_words = {
            word.replace("'", "").replace("’", "")
            for word in NAME_WORD_PATTERN.findall(nlp.clean(query))
            if len(word.replace("'", "").replace("’", "")) >= 4
        }
        author_matches = []
        for author in self.books["author"].dropna().unique():
            author_words = {
                word.replace("'", "").replace("’", "")
                for word in NAME_WORD_PATTERN.findall(nlp.clean(author))
                if len(word.replace("'", "").replace("’", "")) >= 4
            }
            matched_words = query_name_words & author_words
            if matched_words:
                author_matches.append((len(matched_words), sum(map(len, matched_words)), author))
        matched_author = max(author_matches, default=(0, 0, None))[2]

        entities = {"book_title": matched_title, "author": matched_author}
        student = STUDENT_ID_PATTERN.search(query)
        if student:
            entities["student_id"] = int(student.group(1))
        return entities

    @staticmethod
    def _records(frame: pd.DataFrame) -> list[dict]:
        if frame.empty:
            return []
        return frame.where(pd.notna(frame), None).to_dict(orient="records")

    def answer(self, query: str, conn, student_id: int | None = None) -> dict:
        """Answer one query using AI modules and ordinary database services."""
        query = query.strip()
        if not query:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "message": "Please enter a question.",
                "data": [],
                "entities": {},
            }

        intent, confidence = self.classify(query)
        entities = self.extract_entities(query)
        if confidence < CHATBOT_MIN_CONFIDENCE:
            return {
                "intent": intent,
                "confidence": confidence,
                "message": "I am not sure what you mean. Try asking about a title, due dates, recommendations, search, or demand.",
                "data": [],
                "entities": entities,
            }

        if intent == "availability":
            title = entities["book_title"]
            if title is None:
                return self._response(
                    intent, confidence, "Please include a catalog title, for example: Is 'Clean Code' available?",
                    [], entities,
                )
            book = self.books[self.books["title"] == title].iloc[0]
            row = db.availability_df(conn).set_index("book_id").loc[int(book["book_id"])]
            count = int(row["available"])
            message = (
                f"Yes. {title} has {count} available cop{'y' if count == 1 else 'ies'}."
                if count > 0 else f"No copies of {title} are currently available."
            )
            return self._response(intent, confidence, message, [{
                "book_id": int(book["book_id"]), "title": title, "available": count,
            }], entities)

        if intent in {"due_soon", "overdue"}:
            as_of = db.today(conn)
            student_filter = student_id
            if intent == "overdue":
                result = db.overdue_loans_df(conn, as_of=as_of, student_id=student_filter)
                message = f"Found {len(result)} overdue loan(s)."
            else:
                if "next week" in query.lower():
                    start = as_of + timedelta(days=7 - as_of.weekday())
                    end = start + timedelta(days=6)
                else:
                    start = as_of
                    end = as_of + timedelta(days=6 - as_of.weekday())
                result = db.due_loans_df(conn, start=start, end=end, student_id=student_filter)
                message = f"Found {len(result)} loan(s) due between {start} and {end}."
            return self._response(intent, confidence, message, self._records(result), entities)

        if intent == "recommendations":
            if student_id is None:
                return self._response(
                    intent, confidence, "Select a student first so I can personalize recommendations.",
                    [], entities,
                )
            result = self.recommender.recommend(student_id, top_n=5, availability=db.availability_df(conn))
            return self._response(
                intent, confidence, f"Here are {len(result)} recommendations for you.",
                self._records(result), entities,
            )

        if intent == "search":
            result = self.search_engine.search(query, top_n=5, availability=db.availability_df(conn))
            return self._response(
                intent, confidence, f"Found {len(result)} matching title(s).",
                self._records(result), entities,
            )

        if intent == "author_search":
            author = entities["author"]
            if author is None:
                return self._response(
                    intent, confidence, "Please include an author's name so I can find their books.",
                    [], entities,
                )
            result = self.books.loc[
                self.books["author"] == author, ["book_id", "title", "author"]
            ].merge(db.availability_df(conn)[["book_id", "available"]], on="book_id")
            available_only = bool(
                {"available", "availability", "borrow", "borrowable", "stock"}
                .intersection(nlp.raw_words(query))
            )
            if available_only:
                result = result[result["available"] > 0]
                message = f"Found {len(result)} available book(s) by {author}."
            else:
                message = f"Found {len(result)} book(s) by {author}; availability is shown."
            return self._response(
                intent, confidence, message, self._records(result), entities,
            )

        if intent == "forecast":
            result = self.forecaster.forecast(by="genre", horizon_weeks=FORECAST_HORIZON_WEEKS)
            ranked = (
                result.groupby("genre", as_index=False)["predicted_loans"]
                .sum()
                .sort_values("predicted_loans", ascending=False)
                .head(5)
            )
            return self._response(
                intent, confidence, "These genres have the highest predicted borrowing demand.",
                self._records(ranked), entities,
            )

        if intent == "unknown":
            return self._response(
                intent, confidence,
                "I can help with library search, recommendations, availability, due dates, overdue loans, and demand forecasts.",
                [], entities,
            )

        return self._response(
            "help", confidence,
            "I can search the catalog, recommend books, check availability, list due or overdue loans, and forecast demand.",
            [], entities,
        )

    @staticmethod
    def _response(intent: str, confidence: float, message: str, data: list, entities: dict) -> dict:
        return {
            "intent": intent,
            "confidence": confidence,
            "message": message,
            "data": data,
            "entities": entities,
        }
