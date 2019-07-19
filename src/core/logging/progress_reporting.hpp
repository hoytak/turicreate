#ifndef TURI_PROGRESS_REPORTING_HPP_
#define TURI_PROGRESS_REPORTING_HPP_


#include <core/logging/logger.hpp>
#include <mutex>

namespace turi { 

  class progress_context; 

  /**   A simple progress reporting mechanism.
   *    
   *
   *
   */
  class progress_reporter { 
    public:

    progress_reporter(const std::string& task_name, const std::string& task_description, size_t num_work_units); 

    ~progress_reporter();

    size_t increment_completed_units();
    size_t report_work_completion(size_t work_unit_count, size_t current_num_work_units); 

    const std::string& name() const { return m_name; } 
    const std::string& description() const { return m_description; } 


    /** Marks the current task as being completed.  
     *
     *  Also called on descruction. 
     *
     */
    void complete();

  private:
    std::string m_name;
    std::string m_description;
    std::shared_ptr<progress_context> m_owning_context; 

    std::once_flag m_completion_reported; 
  }; 


  class progress_context { 
    public: 

  /** Set a function to be called whenever a progress reports  
   *
   */
  size_t set_progress_hook(std::function<void(const std::string&, size_t, size_t)>&& hook);


  size_t set_task_enter_hook(std::function<void(const std::string&, const std::string&, size_t)>&& hook);


  size_t set_task_completion_hook(std::function<void(const std::string&)>&& hook); 

  std::shared_ptr<progress_reporter> enter_task(const std::string& task_name, const std::string& task_description, size_t num_work_units);


    private: 

  friend class progress_reporter; 

  void _task_completed(const progress_reporter* reportee);



  static std::vector<std::weak_ptr<progress_reporter> > _current_progress_stack;

  static std::vector<std::pair<size_t, std::function<void(const std::string&, size_t, size_t)> > > progress_report_hooks; 
  static std::vector<std::pair<size_t, std::function<void(const std::string&, const std::string&, size_t)> > > task_enter_hooks;
  static std::vector<std::pair<size_t, std::function<void(const std::string&)> > > task_completion_hooks; 





  }





  // Returns the global progress manager.
  static const std::shared_ptr<progress_context>& global_progress_context();



}

#endif
