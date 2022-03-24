# TODO: Write tests
class MaleEvalClassifier:
    """
    Identifies whether any feature with `feature_text`="Male" is present in a given text
    """
    MALE_INDICATORS = {'MAN', 'M', 'MALE', 'GUY', 'DUDE', 'BOY', 'GENTLEMAN', 'GENT', 'HIS', 'HIM', 'HE'}

    def predict(self, X: str) -> int:
        """
        Returns 1 if `X` has a "Female" feature present in it, 0 otherwise
        """
        cleaned = list(map(lambda s: s.upper(), X.split()))
        return int(any(mi in cleaned for mi in self.MALE_INDICATORS))

class FemaleEvalClassifier:
    """
    Identifies whether any feature with `feature_text`="Female" is present in a given text
    """
    FEMALE_INDICATORS = {'WOMAN', 'W', 'SHE', 'LADY', 'FEMALE', 'HER', 'HERS'}

    def predict(self, X: str) -> int:
        """
        Returns 1 if `X` has a "Female" feature present in it, 0 otherwise
        """
        cleaned = list(map(lambda s: s.upper(), X.split()))
        return int(any(mi in cleaned for mi in self.FEMALE_INDICATORS))
