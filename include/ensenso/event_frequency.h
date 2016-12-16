

#ifndef _ENSENSO_EVENT_FREQ_
#define _ENSENSO_EVENT_FREQ_

#include <pcl/common/time.h>
#include <queue>

/** \brief A helper class to measure frequency of a certain event.
 *
 * To use this class create an instance and call event() function every time
 * the event in question occurs. The estimated frequency can be retrieved
 * with getFrequency() function.
 *
 * \author Sergey Alexandrov
 * \ingroup common
 */

namespace ensenso
{
  
/** \brief A helper class to measure frequency of a certain event.
   *
   * To use this class create an instance and call event() function every time
   * the event in question occurs. The estimated frequency can be retrieved
   * with getFrequency() function.
   *
   * \author Sergey Alexandrov
   * \ingroup common
   *
   *
   *
   *
   * ported from pcl 1.8.
   * commit 2ca296bdd0518282fb6254ea860735fedd5306e1
   * 
   * luis galup, 141110
   */
  class EventFrequency
  {

  public:

    /** \brief Constructor.
     *
     * \param[i] window_size number of most recent events that are
     * considered in frequency estimation (default: 30) */
    EventFrequency (size_t window_size = 30)
      : window_size_ (window_size)
    {
      stop_watch_.reset ();
    }

    /** \brief Notifies the class that the event occured. */
    void event ()
    {
      event_time_queue_.push (stop_watch_.getTimeSeconds ());
      if (event_time_queue_.size () > window_size_)
        event_time_queue_.pop ();
    }

    /** \brief Retrieve the estimated frequency. */
    double
      getFrequency () const
    {
      if (event_time_queue_.size () < 2)
        return (0.0);
      return ((event_time_queue_.size () - 1) /
              (event_time_queue_.back () - event_time_queue_.front ()));
    }

    /** \brief Reset frequency computation. */
    void reset ()
    {
      stop_watch_.reset ();
      event_time_queue_ = std::queue<double> ();
    }

  private:

    pcl::StopWatch stop_watch_;
    std::queue<double> event_time_queue_;
    const size_t window_size_;

  };

}

#endif
