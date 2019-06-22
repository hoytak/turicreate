#ifndef TURI_PROGRESS_REPORTING_HPP_
#define TURI_PROGRESS_REPORTING_HPP_


#include <
#include <mutex>

namespace turi { 


  class progress_reporter { 
    public:


    progress_reporter(const std::string& task_name, size_t num_work_units); 
    progress_reporter(std::shared_ptr<progress_reporter> parent_task, const std::string& subtask_name, size_t num_work_units); 
    ~progress_reporter();



    size_t increment_completed_units();
    size_t report_work_completion(size_t work_unit_count, size_t current_num_work_units); 


    /** Marks the current task as being completed.  
     *
     *  Also called on descruction. 
     *
     */
    void complete();

  
    std::shared_ptr<progress_reporter> enter_subtask(const std::string& task_name, const std::string& task_description, size_t num_work_units);





  private:
    std::string m_name;
    std::string m_description;
    std::shared_ptr<progress_reporter> parent;

    std::once_flag m_completion_reported; 
  }; 


  /** Set a function to be called on progress updating.  
   *
   */
  size_t set_progress_hook(std::function<void(const std::string&, size_t, size_t)>&& hook);

  std::shared_ptr<progress_reporter> enter_task(const std::string& task_name, const std::string& task_description, size_t num_work_units);


  



}

#endif
