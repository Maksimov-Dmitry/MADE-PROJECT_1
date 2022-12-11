import pandas as pd


class RecommendationArticles:
    def __init__(self, clustering, co_authors, db, authors_articles, top_n):
        self.clustering = clustering
        self.co_authors = co_authors
        self.db = db
        self.authors_articles = authors_articles
        self.top_n = top_n
        self.columns = ['title', 'year', 'doi']

    def predict(self, text=None, author=None):
        if text and author:
            result = self._get_prediction_text_author(text, author, self.top_n, self.db)
            if len(result) < self.top_n:
                if len(result) == 0:
                    new_db = self.db
                else:
                    new_db = self.db[~self.db._id.isin(result._id)]
                result_text = self._get_prediction_text(text, self.top_n - len(result), new_db)
                result = pd.concat([result, result_text], ignore_index=True)
        elif text:
            result = self._get_prediction_text(text, self.top_n, self.db)
        else:
            result = self._get_prediction_author(text, self.top_n, self.db)
        if result.empty:
            return result.to_json(orient='records')
        return result[self.columns].to_json(orient='records')
                
    def _get_prediction_text_author(self, text, author, top_n, db):
        topic = self.clustering.predict([text])[0][0]
        co_authors = self.co_authors.predict(author)
        result = pd.DataFrame()
        if co_authors and topic:
            for co_author in co_authors:
                if co_author in self.authors_articles:
                    temp_df = db.loc[db._id.isin(self.authors_articles[co_author]) & db.topic.eq(topic)]
                    result = pd.concat([result, temp_df], ignore_index=True)
        if result.empty:
            return result
        return result.nlargest(top_n, 'n_cited')

    def _get_prediction_text(self, text, top_n, db):
        topic = self.clustering.predict([text])[0][0]
        result = db.loc[db.topic.eq(topic)]
        if result.empty:
            return result
        return result.nlargest(top_n, 'n_cited')

    def _get_prediction_author(self, author, top_n, db):
        co_authors = self.co_authors.predict(author)
        result = pd.DataFrame()
        for co_author in co_authors:
            if co_author in self.authors_articles:
                temp_df = db.loc[db._id.isin(self.authors_articles[co_author])]
                result = pd.concat([result, temp_df], ignore_index=True)
        if result.empty:
            return result
        return result.nlargest(top_n, 'n_cited')