#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/rendering/DynamicLines.hh>
#include <gazebo/rendering/Scene.hh>
#include <gazebo/rendering/Visual.hh>
#include <gazebo/rendering/ogre_gazebo.h>

namespace smart_wheelchair_gazebo
{
// LaserVisual uses DynamicLines, which Visual::SetTransparency does not handle.
// Clone its materials so only this sensor's GUI geometry becomes faint.
class FaintLidarVisualPlugin : public gazebo::VisualPlugin
{
 public:
  void Load(gazebo::rendering::VisualPtr visual, sdf::ElementPtr sdf) override
  {
    scene = visual->GetScene();
    laserName = visual->GetParent()->Name() + "::" +
        sdf->Get<std::string>("sensor") + "_GUIONLY_laser_vis";
    connection = gazebo::event::Events::ConnectPreRender([this]() { Update(); });
  }

 private:
  void Update()
  {
    const auto laser = scene->GetVisual(laserName);
    if (!laser) return;
    auto node = laser->GetSceneNode();
    for (unsigned int i = 0; i < node->numAttachedObjects(); ++i)
    {
      auto line = dynamic_cast<gazebo::rendering::DynamicLines *>(node->getAttachedObject(i));
      if (!line) continue;
      const auto original = line->getMaterial();
      if (original->getName().find("Gazebo/") != 0) continue;
      const auto name = "SmartWheelChair/FaintLidar/" + original->getName();
      auto material = Ogre::MaterialManager::getSingleton().getByName(name);
      if (material.isNull())
      {
        material = original->clone(name);
        for (unsigned int t = 0; t < material->getNumTechniques(); ++t)
          for (unsigned int p = 0; p < material->getTechnique(t)->getNumPasses(); ++p)
          {
            auto pass = material->getTechnique(t)->getPass(p);
            auto diffuse = pass->getDiffuse();
            diffuse.a = 0.035f;
            pass->setDiffuse(diffuse);
            pass->setSceneBlending(Ogre::SBT_TRANSPARENT_ALPHA);
            pass->setDepthWriteEnabled(false);
            // Gazebo's laser scripts override diffuse alpha with manual alpha.
            for (unsigned int u = 0; u < pass->getNumTextureUnitStates(); ++u)
              pass->getTextureUnitState(u)->setAlphaOperation(
                  Ogre::LBX_SOURCE1, Ogre::LBS_MANUAL, Ogre::LBS_CURRENT, 0.035f);
          }
      }
      GZ_OGRE_SET_MATERIAL_BY_NAME(line, name);
    }
  }

  gazebo::rendering::ScenePtr scene;
  std::string laserName;
  gazebo::event::ConnectionPtr connection;
};

GZ_REGISTER_VISUAL_PLUGIN(FaintLidarVisualPlugin)
}  // namespace smart_wheelchair_gazebo
