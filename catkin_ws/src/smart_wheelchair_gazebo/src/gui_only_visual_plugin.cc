#include <gazebo/common/Plugin.hh>
#include <gazebo/rendering/Visual.hh>

namespace smart_wheelchair_gazebo
{
// Gazebo 11 Visual messages have no visibility-mask field. Loading this native
// visual plugin applies the mask before the scene renders the new geometry.
class GuiOnlyVisualPlugin : public gazebo::VisualPlugin
{
 public:
  void Load(gazebo::rendering::VisualPtr visual, sdf::ElementPtr) override
  {
    visual->SetVisibilityFlags(GZ_VISIBILITY_GUI);
  }
};

GZ_REGISTER_VISUAL_PLUGIN(GuiOnlyVisualPlugin)
}  // namespace smart_wheelchair_gazebo
