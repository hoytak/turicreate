
/**
 * Fill the ml_data_row with an EigenVector using reference encoding for
 * categorical variables. Here, the 0"th" category is used as the reference
 * category.
 *
 * [in,out] An ml_data_row_reference object from which we are reading.
 * [in,out] An eigen row expression (could be a sparse, dense, or row of a matrix)
 */
template <typename RowExpr>
GL_HOT_INLINE_FLATTEN inline void
supervised_learning_base::fill_row_expression(
    const ml_data_row_reference& row_ref, RowExpr&& x) const {

  DASSERT_EQ(x.size(), m_row_dimension); 
  x.zeros();
  size_t offset = 0;

  row_ref.unpack(

      // The function to write out the data to x.
      [&](ml_column_mode mode, size_t column_index, size_t feature_index,
          double value, size_t index_size, size_t index_offset)
          GL_HOT_INLINE_FLATTEN {

            if (UNLIKELY(feature_index >= index_size)) return;

            // Decrement if it isn't the reference category.
            size_t idx = offset + feature_index;
            if (m_use_reference_encoding && mode == ml_column_mode::CATEGORICAL)) {
              if (feature_index != 0) {
                idx -= 1;
              } else {
                return;
              }
            }

            DASSERT_GE(idx, 0);
            x(idx) = value;

          },

      /**
       * The function to advance the offset, called after each column
       *  is finished.
       */
      [&](ml_column_mode mode, size_t column_index, size_t index_size)
          GL_HOT_INLINE_FLATTEN {
            offset += (index_size - (mode == ml_column_mode::CATEGORICAL ? 1 : 0));
          });
}

