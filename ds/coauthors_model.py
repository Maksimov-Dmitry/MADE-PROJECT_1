class CoAuthors:
    def __init__(self, association_rules, top_n):
        self.association_rules = association_rules
        self.top_n = top_n

    def predict(self, author):
        return [
            coauthor
            for coauthors in self.association_rules[self.association_rules.antecedents.eq({author})].nlargest(self.top_n, 'lift').consequents
            for coauthor in coauthors
        ]