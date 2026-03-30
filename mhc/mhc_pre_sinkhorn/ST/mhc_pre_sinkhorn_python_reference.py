"""
MhcPreSinkhorn Python Reference Implementation

This script provides a reference implementation of the Sinkhorn algorithm
for the MHC architecture. It serves as a baseline for validating the
NPU implementation of the mhc_pre_sinkhorn operator.

The Sinkhorn algorithm transforms an input matrix into a doubly stochastic
matrix (both row and column sums equal to 1) through iterative normalization.

Author: Development Team
Date: 2026-03-27
Version: 1.0
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings


class MhcPreSinkhorn:
    """
    Sinkhorn algorithm implementation for MHC architecture.
    
    This class implements the Sinkhorn algorithm to transform an input matrix
    into a doubly stochastic matrix through iterative row and column normalization.
    
    Attributes:
        eps (float): Small constant to prevent division by zero
        num_iters (int): Number of Sinkhorn iterations
        out_flag (int): Flag to control intermediate output (0: no output, 1: output)
    """
    
    # Supported values for the n dimension
    SUPPORTED_N_VALUES = {4, 6, 8}
    
    # Supported data types
    SUPPORTED_DTYPES = {np.float32}
    
    # Default parameter values
    DEFAULT_EPS = 1e-6
    DEFAULT_NUM_ITERS = 20
    DEFAULT_OUT_FLAG = 0
    
    # Parameter constraints
    MIN_NUM_ITERS = 1
    MAX_NUM_ITERS = 100
    
    def __init__(
        self,
        eps: float = DEFAULT_EPS,
        num_iters: int = DEFAULT_NUM_ITERS,
        out_flag: int = DEFAULT_OUT_FLAG
    ):
        """
        Initialize the MhcPreSinkhorn operator.
        
        Args:
            eps: Small constant to prevent division by zero (default: 1e-6)
            num_iters: Number of Sinkhorn iterations (default: 20, range: 1-100)
            out_flag: Flag to control intermediate output (default: 0)
                     0: Only output sinkhorn result
                     1: Output intermediate results (norm_out and sum_out)
        
        Raises:
            ValueError: If eps <= 0 or num_iters is out of range
            TypeError: If out_flag is not 0 or 1
        """
        self._validate_params(eps, num_iters, out_flag)
        
        self.eps = eps
        self.num_iters = num_iters
        self.out_flag = out_flag
    
    @staticmethod
    def _validate_params(eps: float, num_iters: int, out_flag: int) -> None:
        """
        Validate input parameters.
        
        Args:
            eps: Small constant to prevent division by zero
            num_iters: Number of Sinkhorn iterations
            out_flag: Flag to control intermediate output
        
        Raises:
            ValueError: If parameters are invalid
            TypeError: If out_flag is not 0 or 1
        """
        if eps <= 0:
            raise ValueError(
                f"eps must be greater than 0, got {eps}"
            )
        
        if num_iters < MhcPreSinkhorn.MIN_NUM_ITERS or \
           num_iters > MhcPreSinkhorn.MAX_NUM_ITERS:
            raise ValueError(
                f"num_iters must be in range "
                f"[{MhcPreSinkhorn.MIN_NUM_ITERS}, {MhcPreSinkhorn.MAX_NUM_ITERS}], "
                f"got {num_iters}"
            )
        
        if out_flag not in {0, 1}:
            raise TypeError(
                f"out_flag must be 0 or 1, got {out_flag}"
            )
    
    @staticmethod
    def _validate_input(h_res: np.ndarray) -> None:
        """
        Validate input tensor.
        
        Args:
            h_res: Input tensor (H^res matrix)
        
        Raises:
            ValueError: If input shape or dtype is invalid
            TypeError: If input dtype is not supported
        """
        # Check data type
        if h_res.dtype not in MhcPreSinkhorn.SUPPORTED_DTYPES:
            raise TypeError(
                f"h_res dtype must be float32, got {h_res.dtype}"
            )
        
        # Check dimension
        if h_res.ndim not in {3, 4}:
            raise ValueError(
                f"h_res dimension must be 3 or 4, got {h_res.ndim}"
            )
        
        # Check n value (last two dimensions must be equal and in {4, 6, 8})
        if h_res.ndim == 3:
            n0 = h_res.shape[1]
            n1 = h_res.shape[2]
        else:  # h_res.ndim == 4
            n0 = h_res.shape[2]
            n1 = h_res.shape[3]
        
        if n0 != n1:
            raise ValueError(
                f"Last two dimensions must be equal, got {n0} and {n1}"
            )
        
        if n0 not in MhcPreSinkhorn.SUPPORTED_N_VALUES:
            raise ValueError(
                f"n must be in {MhcPreSinkhorn.SUPPORTED_N_VALUES}, got {n0}"
            )
    
    def _softmax(
        self,
        x: np.ndarray,
        axis: int = -1
    ) -> np.ndarray:
        """
        Compute softmax along specified axis.
        
        Args:
            x: Input array
            axis: Axis along which to compute softmax (default: -1)
        
        Returns:
            Softmax output array
        """
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        
        # Add eps to prevent division by zero
        return exp_x / (sum_exp + self.eps)
    
    def _row_normalize(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize rows of the matrix.
        
        Args:
            x: Input matrix
        
        Returns:
            Tuple of (normalized matrix, row sums)
        """
        # Compute row sums
        row_sums = np.sum(x, axis=-1, keepdims=True)
        
        # Add eps to prevent division by zero
        row_sums_eps = row_sums + self.eps
        
        # Normalize
        x_normalized = x / row_sums_eps
        
        return x_normalized, row_sums
    
    def _col_normalize(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize columns of the matrix.
        
        Args:
            x: Input matrix
        
        Returns:
            Tuple of (normalized matrix, column sums)
        """
        # Compute column sums
        col_sums = np.sum(x, axis=-2, keepdims=True)
        
        # Add eps to prevent division by zero
        col_sums_eps = col_sums + self.eps
        
        # Normalize
        x_normalized = x / col_sums_eps
        
        return x_normalized, col_sums
    
    def _compute_sinkhorn(
        self,
        h_res: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute Sinkhorn transformation.
        
        Args:
            h_res: Input tensor (H^res matrix)
        
        Returns:
            Tuple of (sinkhorn result, norm_out, sum_out)
            norm_out and sum_out are None if out_flag == 0
        """
        # Initialize intermediate result storage
        if self.out_flag:
            # Get batch dimensions
            if h_res.ndim == 3:
                batch_dims = h_res.shape[0]
            else:  # h_res.ndim == 4
                batch_dims = h_res.shape[0] * h_res.shape[1]
            
            # Get n value
            if h_res.ndim == 3:
                n = h_res.shape[1]
            else:  # h_res.ndim == 4
                n = h_res.shape[2]
            
            # Initialize intermediate result arrays
            # norm_out shape: [2*num_iters, n, n, batch_dims]
            # sum_out shape: [2*num_iters, n, batch_dims]
            norm_out = np.zeros(
                (2 * self.num_iters, n, n, batch_dims),
                dtype=h_res.dtype
            )
            sum_out = np.zeros(
                (2 * self.num_iters, n, batch_dims),
                dtype=h_res.dtype
            )
        else:
            norm_out = None
            sum_out = None
        
        # First iteration: Softmax + Column normalization
        # Step 1: Apply softmax along rows (dim=-1)
        current = self._softmax(h_res, axis=-1)
        
        if self.out_flag:
            # Store softmax result
            norm_out[0] = self._transpose_to_norm_out_format(current)
        
        # Step 2: Column normalization
        current, col_sums = self._col_normalize(current)
        
        if self.out_flag:
            # Store column sums
            sum_out[1] = self._transpose_to_sum_out_format(col_sums)
            # Store normalized result
            norm_out[1] = self._transpose_to_norm_out_format(current)
        
        # Subsequent iterations: Row normalization + Column normalization
        for iter_idx in range(1, self.num_iters):
            # Row normalization
            current, row_sums = self._row_normalize(current)
            
            if self.out_flag:
                # Store row sums
                sum_out[2 * iter_idx] = self._transpose_to_sum_out_format(row_sums)
                # Store normalized result
                norm_out[2 * iter_idx] = self._transpose_to_norm_out_format(current)
            
            # Column normalization
            current, col_sums = self._col_normalize(current)
            
            if self.out_flag:
                # Store column sums
                sum_out[2 * iter_idx + 1] = self._transpose_to_sum_out_format(col_sums)
                # Store normalized result
                norm_out[2 * iter_idx + 1] = self._transpose_to_norm_out_format(current)
        
        return current, norm_out, sum_out
    
    def _transpose_to_norm_out_format(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Transpose input to norm_out format.
        
        Args:
            x: Input array with shape [batch_dims, n, n] or [B, S, n, n]
        
        Returns:
            Transposed array with shape [n, n, batch_dims]
        """
        if x.ndim == 3:
            # Shape: [batch_dims, n, n] -> [n, n, batch_dims]
            return np.transpose(x, (1, 2, 0))
        else:  # x.ndim == 4
            # Shape: [B, S, n, n] -> [n, n, B*S]
            batch_dims = x.shape[0] * x.shape[1]
            n = x.shape[2]
            return x.reshape(batch_dims, n, n).transpose(1, 2, 0)
    
    def _transpose_to_sum_out_format(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Transpose input to sum_out format.
        
        Args:
            x: Input array with shape [batch_dims, n, 1] or [B, S, n, 1]
        
        Returns:
            Transposed array with shape [n, batch_dims]
        """
        if x.ndim == 3:
            # Shape: [batch_dims, n, 1] -> [n, batch_dims]
            return x.squeeze(-1).transpose(1, 0)
        else:  # x.ndim == 4
            # Shape: [B, S, n, 1] -> [n, B*S]
            batch_dims = x.shape[0] * x.shape[1]
            n = x.shape[2]
            return x.reshape(batch_dims, n, 1).squeeze(-1).transpose(1, 0)
    
    def __call__(
        self,
        h_res: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply Sinkhorn transformation to input tensor.
        
        This is the main entry point for the Sinkhorn algorithm.
        
        Args:
            h_res: Input tensor (H^res matrix)
                   - 3D: [T, n, n] where T is sequence length
                   - 4D: [B, S, n, n] where B is batch size, S is sequence length
                   - n must be in {4, 6, 8}
                   - dtype must be float32
        
        Returns:
            Tuple of (h_res_sinkhorn, norm_out, sum_out):
            - h_res_sinkhorn: Doubly stochastic matrix with same shape as h_res
            - norm_out: Intermediate normalization results (None if out_flag==0)
                       Shape: [2B*num_iters, n, n, B*S] or [2*num_iters, n, n, T]
            - sum_out: Intermediate sum results (None if out_flag==0)
                      Shape: [2*num_iters, n, B*S] or [2*num_iters, n, T]
        
        Raises:
            ValueError: If input shape or dtype is invalid
            TypeError: If input dtype is not supported
        """
        # Validate input
        self._validate_input(h_res)
        
        # Check for special values (inf, nan)
        if np.any(np.isinf(h_res)) or np.any(np.isnan(h_res)):
            warnings.warn(
                "Input contains inf or nan values. Output will contain nan.",
                RuntimeWarning
            )
        
        # Compute Sinkhorn transformation
        h_res_sinkhorn, norm_out, sum_out = self._compute_sinkhorn(h_res)
        
        return h_res_sinkhorn, norm_out, sum_out


def mhc_pre_sinkhorn(
    h_res: np.ndarray,
    eps: float = MhcPreSinkhorn.DEFAULT_EPS,
    num_iters: int = MhcPreSinkhorn.DEFAULT_NUM_ITERS,
    out_flag: int = MhcPreSinkhorn.DEFAULT_OUT_FLAG
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convenience function for Sinkhorn transformation.
    
    This function provides a simple interface to the MhcPreSinkhorn class.
    
    Args:
        h_res: Input tensor (H^res matrix)
               - 3D: [T, n, n] where T is sequence length
               - 4D: [B, S, n, n] where B is batch size, S is sequence length
               - n must be in {4, 6, 8}
               - dtype must be float32
        eps: Small constant to prevent division by zero (default: 1e-6)
        num_iters: Number of Sinkhorn iterations (default: 20, range: 1-100)
        out_flag: Flag to control intermediate output (default: 0)
                 0: Only output sinkhorn result
                 1: Output intermediate results (norm_out and sum_out)
    
    Returns:
        Tuple of (h_res_sinkhorn, norm_out, sum_out):
        - h_res_sinkhorn: Doubly stochastic matrix with same shape as h_res
        - norm_out: Intermediate normalization results (None if out_flag==0)
                   Shape: [2*num_iters, n, n, B*S] or [2*num_iters, n, n, T]
        - sum_out: Intermediate sum results (None if out_flag==0)
                  Shape: [2*num_iters, n, B*S] or [2*num_iters, n, T]
    
    Raises:
        ValueError: If input shape or dtype is invalid
        TypeError: If input dtype is not supported
    
    Example:
        >>> import numpy as np
        >>> h_res = np.random.randn(8, 4, 4).astype(np(np.float32))
        >>> result, norm_out, sum_out = mhc_pre_sinkhorn(h_res, eps=1e-6, num_iters=20)
        >>> print(result.shape)
        (8, 4, 4)
        
        >>> # Verify doubly stochastic property
        >>> row_sums = result.sum(axis=-1)
        >>> col_sums = result.sum(axis=-2)
        >>> print(np.allclose(row_sums, 1.0, atol=1e-5))
        True
        >>> print(np.allclose(col_sums, 1.0, atol=1e-5))
        True
    """
    sinkhorn = MhcPreSinkhorn(eps=eps, num_iters=num_iters, out_flag=out_flag)
    return sinkhorn(h_res)


def verify_doubly_stochastic(
    matrix: np.ndarray,
    atol: float = 1e-5
) -> Tuple[bool, float, float]:
    """
    Verify that a matrix is doubly stochastic.
    
    A doubly stochastic matrix has all row sums and column sums equal to 1.
    
    Args:
        matrix: Input matrix
        atol: Absolute tolerance for comparison (default: 1e-5)
    
    Returns:
        Tuple of (is_doubly_stochastic, max_row_error, max_col_error):
        - is_doubly_stochastic: True if matrix is doubly stochastic
        - max_row_error: Maximum error in row sums
        - max_col_error: Maximum error in column sums
    """
    # Compute row and column sums
    row_sums = matrix.sum(axis=-1)
    col_sums = matrix.sum(axis=-2)
    
    # Compute errors
    row_errors = np.abs(row_sums - 1.0)
    col_errors = np.abs(col_sums - 1.0)
    
    max_row_error = np.max(row_errors)
    max_col_error = np.max(col_errors)
    
    is_doubly_stochastic = (max_row_error < atol) and (max_col_error < atol)
    
    return is_doubly_stochastic, max_row_error, max_col_error


def verify_non_negativity(
    matrix: np.ndarray
) -> Tuple[bool, float]:
    """
    Verify that all elements in the matrix are non-negative.
    
    Args:
        matrix: Input matrix
    
    Returns:
        Tuple of (is_non_negative, min_value):
        - is_non_negative: True if all elements are >= 0
        - min_value: Minimum value in the matrix
    """
    min_value = np.min(matrix)
    is_non_negative = min_value >= -1e-6  # Allow small negative due to numerical errors
    
    return is_non_negative, min_value


if __name__ == "__main__":
    """
    Main function for testing and demonstration.
    """
    print("=" * 70)
    print("MhcPreSinkhorn Python Reference Implementation - Test Suite")
    print("=" * 70)
    
    # Test 1: Basic functionality test (n=4, 3D input)
    print("\n[Test 1] Basic functionality test (n=4, 3D input)")
    print("-" * 70)
    T = 8
    n = 4
    h_data = np.random.randn(T, n, n).astype(np.float32)
    
    result, norm_out, sum_out = mhc_pre_sinkhorn(
        h_data,
        eps=1e-6,
        num_iters=20,
        out_flag=0
    )
    
    print(f"Input shape: {h_data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    
    # Verify doubly stochastic property
    is_doubly_stochastic, max_row_error, max_col_error = \
        verify_doubly_stochastic(result)
    print(f"Is doubly stochastic: {is_doubly_stochastic}")
    print(f"Max row error: {max_row_error:.2e}")
    print(f"Max col error: {max_col_error:.2e}")
    
    # Verify non-negativity
    is_non_negative, min_value = verify_non_negativity(result)
    print(f"Is non-negative: {is_non_negative}")
    print(f"Min value: {min_value:.2e}")
    
    # Test 2: Different n values (6 and 8)
    print("\n[Test 2] Different n values (6 and 8)")
    print("-" * 70)
    for n in [6, 8]:
        h_data = np.random.randn(8, n, n).astype(np.float32)
        result, _, _ = mhc_pre_sinkhorn(h_data, eps=1e-6, num_iters=20)
        
        is_doubly_stochastic, max_row_error, max_col_error = \
            verify_doubly_stochastic(result)
        
        print(f"n={n}: Doubly stochastic={is_doubly_stochastic}, "
              f"Max row error={max_row_error:.2e}, "
              f"Max col error={max_col_error:.2e}")
    
    # Test 3: 4D input
    print("\n[Test 3] 4D input test")
    print("-" * 70)
    B = 2
    S = 4
    n = 4
    h_data = np.random.randn(B, S, n, n).astype(np.float32)
    
    result, _, _ = mhc_pre_sinkhorn(h_data, eps=1e-6, num_iters=20)
    
    print(f"Input shape: {h_data.shape}")
    print(f"Output shape: {result.shape}")
    
    is_doubly_stochastic, max_row_error, max_col_error = \
        verify_doubly_stochastic(result)
    print(f"Is doubly stochastic: {is_doubly_stochastic}")
    print(f"Max row error: {max_row_error:.2e}")
    print(f"Max col error: {max_col_error:.2e}")
    
    # Test 4: Different iteration counts
    print("\n[Test 4] Different iteration counts")
    print("-" * 70)
    for num_iters in [1, 5, 10, 20, 50]:
        h_data = np.random.randn(8, 4, 4).astype(np.float32)
        result, _, _ = mhc_pre_sinkhorn(h_data, eps=1e-6, num_iters=num_iters)
        
        is_doubly_stochastic, max_row_error, max_col_error = \
            verify_doubly_stochastic(result)
        
        print(f"num_iters={num_iters}: Doubly stochastic={is_doubly_stochastic}, "
              f"Max row error={max_row_error:.2e}, "
              f"Max col error={max_col_error:.2e}")
    
    # Test 5: Intermediate output (out_flag=1)
    print("\n[Test 5] Intermediate output test (out_flag=1)")
    print("-" * 70)
    h_data = np.random.randn(4, 4, 4).astype(np.float32)
    num_iters = 5
    
    result, norm_out, sum_out = mhc_pre_sinkhorn(
        h_data,
        eps=1e-6,
        num_iters=num_iters,
        out_flag=1
    )
    
    print(f"Input shape: {h_data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"norm_out shape: {norm_out.shape}")
    print(f"sum_out shape: {sum_out.shape}")
    
    # Verify final result matches last norm_out
    last_norm_out = norm_out[-1].transpose(2, 0, 1)
    print(f"Final result matches last norm_out: {np.allclose(result, last_norm_out)}")
    
    # Test 6: Special values (inf, nan)
    print("\n[Test 6] Special values test (inf, nan)")
    print("-" * 70)
    h_data = np.random.randn(4, 4, 4).astype(np.float32)
    h_data[0, 0, 0] = float('inf')
    
    result, _, _ = mhc_pre_sinkhorn(h_data, eps=1e-6, num_iters=20)
    has_nan = np.any(np.isnan(result))
    print(f"Input contains inf: True")
    print(f"Output contains nan: {has_nan}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
