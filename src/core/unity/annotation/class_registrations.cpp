#include <core/unity/annotation/class_registrations.hpp>

#include <core/unity/annotation/annotation_base.hpp>
#include <core/unity/annotation/image_classification.hpp>

namespace turi {
namespace annotate {

BEGIN_CLASS_REGISTRATION
REGISTER_CLASS(ImageClassification)
REGISTER_CLASS(annotation_global)
END_CLASS_REGISTRATION

BEGIN_FUNCTION_REGISTRATION
REGISTER_FUNCTION(create_image_classification_annotation, "data",
                  "data_columns", "annotation_column");
END_FUNCTION_REGISTRATION

} // namespace annotate
} // namespace turi
