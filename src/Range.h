#ifndef FVSAND_NVTXRANGE_HPP_
#define FVSAND_NVTXRANGE_HPP_

// C/C++ includes
#include <string>  // for std::string

namespace nvtx
{
/*!
 * \class Range
 *
 * \brief Range is a simple utility class to annotate code.
 *
 *  The NVTXRange class is a simple utility class that can be used in
 *  conjunction with the NVIDIA Tools Extension library to allow developers
 *  to easily mark and annotate code in order to provide additional information
 *  to NVIDIA performance tools, such as, nvprof, nvvp and Nsight.
 *
 * \see https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx
 *
 * \note NVTXRange uses the RAII idiom, consequently the range is started
 *  when the NVTXRange object is instantiated and stopped when the object
 *  goes out of scope.
 *
 * \remark Thanks to Jason Burmark (burmark1@llnl.gov) for his original 
 *  implementation that inspired the implementation of this class.
 *
 * Usage Example:
 * \code
 *
 *    // use scope to auto-start and stop a range
 *    { // begin scope resolution
 *      NVTXRage range ("foo" );
 *      foo();
 *    } // end scope resoltuion
 *
 * \endcode
 *
 */
class Range
{
public:
  /*!
   * \brief Default constructor. Disabled.
   */
  Range() = delete;

  /*!
   * \brief Creates an NVTXRage instance with the given name.
   *
   * \param [in] name the name to associate with the range
   *
   * \pre name.empty() == false
   */
  Range(const std::string& name);

  /*!
   * \brief Destructor.
   */
  ~Range();

private:
  /*!
   * \brief Starts an NVTX range.
   * \note Called by the constructor.
   */
  void start();

  /*!
   * \brief Stops the NVTX range.
   * \note Called by the destructor.
   */
  void stop();

  std::string m_name;
  bool m_active;
};

} /* namespace nvtx */

#endif /* FVSAND_NVTXRANGE_HPP_ */
