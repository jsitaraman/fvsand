#ifndef FVSAND_NVTX_MACROS_HPP_
#define FVSAND_NVTX_MACROS_HPP_

#include "Range.h"

/*!
 * \file
 *
 * \brief Defines NVTX Macros that can be used to annotate functions and
 *  sections of the code. These macros use the NVIDIA Tools Extension library
 *  to provide additional information to NVIDIA performance tools, e.g.,
 *  nvprof, nvvp, Nsight, thereby, facilitate developers in performance
 *  evaluation.
 *
 */

/// \name FVSAND NVTX Macros
///@{

/*!
 * \def FVSAND_NVTX_SECTION
 *
 * \brief The FVSAND_NVTX_SECTION macro is used to annotate sections of code
 *
 * \note In contrast to the FVSAND_NVTX_FUNCTION macro, the FVSAND_NVTX_SECTION
 *   macro is used to annotate sections of code, at a much finer granularity,
 *   within a given function.
 *
 * \warning Variables declared within a given FVSAND_NVTX_SECTION are only defined
 *  within the scope of the FVSAND_NVTX_SECTION.
 *
 * \warning An FVSAND_NVTX_SECTION cannot be called in a nested fashion, i.e.,
 *  within another FVSAND_NVTX_SECTION
 *
 * \note You may have multiple FVSAND_NVTX_SECTION defined within a function and
 *  this macro can be used in conjunction with the FVSAND_NVTX_FUNCTION macro.
 *
 * \Usage Example:
 * \code
 *
 *   void foo( )
 *   {
 *     FVSAND_NVTX_FUNCTION( "foo"" );
 *
 *     // STEP 0: Run kernel A
 *     FVSAND_NVTX_SECTION( "kernelA",
 *
 *        FVSAND_GPU_LAUNCH_FUNC( ... );
 *
 *     ); // END NVTX SECTION for kernel A
 *
 *     // STEP 1: Run kernel B
 *     FVSAND_NVTX_SECTION( "kernelB",
 *
 *        FVSAND_GPU_LAUNCH_FUNC( ... );
 *
 *     ); // END NVTX SECTION for kernel B
 *
 *   }
 * \endcode
 *
 */
#if defined(FVSAND_HAS_GPU) && !defined(FVSAND_FAKE_GPU)
  #define FVSAND_NVTX_SECTION(__name__, ...)             \
    do                                                   \
    {                                                    \
      nvtx::Range r(__name__);                           \
      __VA_ARGS__                                        \
    } while(false)
#else
  #define FVSAND_NVTX_SECTION(__name__, ...)             \
    do                                                   \
    {                                                    \
      __VA_ARGS__                                        \
    } while(false)
#endif

/*!
 * \def FVSAND_NVTX_FUNCTION( name )
 *
 * \brief The FVSAND_NVTX_FUNCTION macro is used to annotate a function.
 * \param [in] name a user-supplied name that will be given to the range.
 *
 * \note Typically, the FVSAND_NVTX_FUNCTION macro is placed in the beginning of
 *  the function to annotate.
 *
 * \warning The FVSAND_NVTX_FUNCTION can be called once within a (function) scope.
 *
 * Usage Example:
 * \code
 *   void foo( )
 *   {
 *     FVSAND_NVTX_FUNCTION( "foo" );
 *     ...
 *   }
 * \endcode
 *
 */
#if defined(FVSAND_HAS_GPU) && !defined(FVSAND_FAKE_GPU)
  #define FVSAND_NVTX_FUNCTION(__name__) nvtx::Range __func_range(__name__)
#else
  #define FVSAND_NVTX_FUNCTION(__name__)
#endif

///@}

#endif /* FVSAND_NVTX_MACROS_HPP_ */
