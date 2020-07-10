from sklearn.preprocessing import LabelEncoder


class MissingLabelEncoder(LabelEncoder):

    __new_class_placeholder = "__unobserved___"

    def fit(self, y):
        if self.__new_class_placeholder in y:
            raise ValueError()
        
        self._true_classes = set(y)
        z = np.concatenate([y, [self.__new_class_placeholder]])
        return super().fit(z)

    def transform(self, y):
        z = np.where(
            np.isin(y, self._true_classes),
            y,
            self.__new_class_placeholder,
        )
        return super().transform(z)