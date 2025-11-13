import math


class PassAtKMetrics:
    """A class for calculating pass@k and pass^k metrics."""

    @staticmethod
    def pass_at_k(n: int, c: int, k: int) -> float:
        """Calculate pass@k metric: probability that at least one of k independent attempts will succeed.

        Args:
            n (int): Total number of attempts/samples
            c (int): Number of correct solutions
            k (int): Number of samples to draw

        Returns:
            float: Pass@k probability (0 to 1)

        Raises:
            ValueError: If invalid parameters are provided
        """
        if n <= 0 or c < 0 or k <= 0:
            raise ValueError("n and k must be positive, c must be non-negative")
        if c > n:
            raise ValueError(
                "Number of correct solutions (c) cannot exceed total attempts (n)"
            )
        if k > n:
            raise ValueError("Sample size (k) cannot exceed total attempts (n)")

        # If all solutions are correct, pass@k = 1
        if c == n:
            return 1.0

        # If no solutions are correct, pass@k = 0
        if c == 0:
            return 0.0

        # Calculate using the complement: 1 - P(all k samples are incorrect)
        # P(all incorrect) = C(n-c, k) / C(n, k)
        try:
            prob_all_incorrect = math.comb(n - c, k) / math.comb(n, k)
            return 1.0 - prob_all_incorrect
        except (ValueError, ZeroDivisionError):
            # Handle edge cases where combinations are invalid
            return 0.0

    @staticmethod
    def pass_power_k(success_rate: float, k: int) -> float:
        """Calculate pass^k metric: probability that an agent would succeed on all k independent attempts.

        Args:
            success_rate (float): Raw success rate on a single attempt (0 to 1)
            k (int): Number of consecutive attempts

        Returns:
            float: Pass^k probability (0 to 1)

        Raises:
            ValueError: If invalid parameters are provided
        """
        if not 0 <= success_rate <= 1:
            raise ValueError("Success rate must be between 0 and 1")
        if k <= 0:
            raise ValueError("k must be positive")

        return success_rate**k

    @staticmethod
    def calculate_success_rate(n: int, c: int) -> float:
        """Calculate raw success rate from total attempts and correct solutions.

        Args:
            n (int): Total number of attempts
            c (int): Number of correct solutions

        Returns:
            float: Success rate (0 to 1)
        """
        if n <= 0:
            raise ValueError("Total attempts (n) must be positive")
        if c < 0:
            raise ValueError("Correct solutions (c) must be non-negative")
        if c > n:
            raise ValueError("Correct solutions (c) cannot exceed total attempts (n)")

        return c / n

    @classmethod
    def calc_pass_k_metrics(cls, attempt_result: list, k: int) -> dict:
        """Calculate pass@k and pass^k metrics for the same data.

        Args:
            attempt_result (list): List containing results of different attempts
            k (int): Number of samples/attempts

        Returns:
            dict: Dictionary containing both metrics and related information
        """
        n = len(attempt_result)
        c = sum(attempt_result)
        success_rate = cls.calculate_success_rate(n, c)
        pass_at_k_value = cls.pass_at_k(n, c, k)
        pass_power_k_value = cls.pass_power_k(success_rate, k)

        return {
            "success_rate": success_rate,
            "pass_at_k": pass_at_k_value,
            "pass_power_k": pass_power_k_value,
            "difference": pass_at_k_value - pass_power_k_value,
            "parameters": {"n": n, "c": c, "k": k},
        }
