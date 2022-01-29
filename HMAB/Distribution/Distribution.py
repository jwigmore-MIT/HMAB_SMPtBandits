

class Distribution(object):
    """ Manipulate Distribution experiments."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This method __init__(self, *args, **kwargs) has to be implemented in the child class inheriting from Distribution.")

    def reset(self, *args, **kwargs):
        """Reset posterior, new experiment."""
        raise NotImplementedError("This method reset(self, *args, **kwargs) has to be implemented in the child class inheriting from Distribution.")

    def sample(self):
        """Sample from the posterior."""
        raise NotImplementedError("This method sample(self) has to be implemented in the child class inheriting from Distribution.")

    def quantile(self, p):
        """p quantile from the posterior."""
        raise NotImplementedError("This method quantile(self, p) has to be implemented in the child class inheriting from Distribution.")

    def getExpVal(self):
        """Mean of the posterior."""
        raise NotImplementedError("This method getExpVal(self) has to be implemented in the child class inheriting from Distribution.")

    def forget(self, obs):
        """Forget last observation (never used)."""
        raise NotImplementedError("This method forget(self, obs) has to be implemented in the child class inheriting from Distribution.")

    def update(self, obs):
        """Update posterior with this observation."""
        raise NotImplementedError("This method update(self, obs) has to be implemented in the child class inheriting from Distribution.")
