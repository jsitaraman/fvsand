#include "Range.h"

// C/C++ includes
#include <cassert>

// CUDA NVTX includes
#if defined(FVSAND_HAS_GPU) && !defined(FVSAND_FAKE_GPU)
  #include <cuda.h>
  #include <nvToolsExt.h>
  #include <nvToolsExtCuda.h>
#endif


namespace nvtx
{

/*!
 * \brief Predefined set of NVTX colors to use with NVTXRange.
 *
 * TODO: Expose this to the application that a color can be set.
 */
enum class Color : uint32_t
{
  BLACK   = 0x00000000
, GREEN   = 0x0000FF00
, LIME    = 0x00BFFF00
, RED     = 0x00FF0000
, BLUE    = 0x000000FF
, YELLOW  = 0x00FFFF00
, CYAN    = 0x0000FFFF
, MAGENTA = 0x00FF00FF
, WHITE   = 0x00FFFFFF
, ORANGE  = 0x00FFA500
, PINK    = 0x00FF69B4
};

Range::Range(const std::string& name) : m_name(name), m_active(false)
{
  assert(!m_name.empty());
  start();
  assert(m_active);
}

//------------------------------------------------------------------------------
Range::~Range() { stop(); }

//------------------------------------------------------------------------------
void Range::start()
{
  assert(!m_active);

#if defined(FVSAND_HAS_GPU) && !defined(FVSAND_FAKE_GPU)
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.category = 0 /* any category */;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(nvtx::Color::GREEN);
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = m_name.c_str();

  nvtxRangePushEx(&eventAttrib);
#endif

  m_active = true;
}

//------------------------------------------------------------------------------
void Range::stop()
{
  if(m_active)
  {
#if defined(FVSAND_HAS_GPU) && !defined(FVSAND_FAKE_GPU)
    nvtxRangePop();
#endif
    m_active = false;
  }
}

} /* namespace nvtx */
