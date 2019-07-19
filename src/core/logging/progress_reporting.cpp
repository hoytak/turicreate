#include <core/logging/logger.hpp>
#include <core/logging/progress_reporting.cpp>
#include <mutex>
#include <parallel/atomic.hpp>

namespace turi { 


  static std::vector<std::weak_ptr<progress_reporter> > _current_progress_stack;
  atomic<size_t> hook_id = 0; 

  static std::vector<std::pair<size_t, std::function<void(const std::string&, size_t, size_t)> > > progress_report_hooks; 
  static std::vector<std::pair<size_t, std::function<void(const std::string&, const std::string&, size_t)> > > task_enter_hooks;
  static std::vector<std::pair<size_t, std::function<void(const std::string&)> > > task_completion_hooks; 



  /** Set a function to be called on progress updating. 
   *
   */
  size_t set_progress_hook(std::function<void(const std::string&, size_t, size_t)>&& hook);


  size_t set_task_enter_hook(std::function<void(const std::string&, size_t)>&& hook);


  size_t set_task_completion_hook(std::function<void(const std::string&)>&& hook); 





  std::shared_ptr<progress_reporter> enter_task(const std::string& task_name, const std::string& task_description, size_t num_work_units) {








  }


  static void task_completed(const progress_reporter* reported) {
     ASSERT_TRUE(_current_progress_stack.back().get() == reported); 


     for(const auto& p : task_completion_hooks) { 



     }



  }
  




  static void 




  class progress_reporter { 
    public:

    progress_reporter(const std::string& task_name, const std::string& task_description, size_t num_work_units); 

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

    std::once_flag m_completion_reported; 
  }; 


  /** Set a function to be called on progress updating.  
   *
   */
  size_t set_progress_hook(std::function<void(const std::string&, size_t, size_t)>&& hook);


  size_t set_task_enter_hook(std::function<void(const std::string&, size_t)>&& hook); 
  size_t set_task_completion_hook(std::function<void(const std::string&, size_t)>&& hook); 

  std::shared_ptr<progress_reporter> enter_task(const std::string& task_name, const std::string& task_description, size_t num_work_units);
}

#endif






static std::vector<std::weak_ptr<progress_reporter

