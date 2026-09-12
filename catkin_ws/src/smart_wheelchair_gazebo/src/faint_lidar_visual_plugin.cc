#include <algorithm>
#include <cmath>
#include <vector>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/rendering/DynamicLines.hh>
#include <gazebo/rendering/Scene.hh>
#include <gazebo/rendering/Visual.hh>
#include <gazebo/rendering/ogre_gazebo.h>

namespace smart_wheelchair_gazebo
{
// LaserVisual uses DynamicLines, which Visual::SetTransparency does not handle.
// GUI-only colors and body occlusion; never modify the sensor scan messages.
class FaintLidarVisualPlugin : public gazebo::VisualPlugin
{
 public:
  void Load(gazebo::rendering::VisualPtr visual, sdf::ElementPtr sdf) override
  {
    scene = visual->GetScene();
    housing = visual;
    const auto sensor = sdf->Get<std::string>("sensor");
    lowerAngle = (sensor == "left_lidar" ? -30. : -170.) * M_PI / 180.;
    upperAngle = (sensor == "left_lidar" ? 170. : 30.) * M_PI / 180.;
    color = sensor == "left_lidar" ? Ogre::ColourValue(.1f, .65f, 1.f, .08f)
                                  : Ogre::ColourValue(.75f, .1f, 1.f, .08f);
    laserName = visual->GetParent()->Name() + "::" +
        sensor + "_GUIONLY_laser_vis";
    connection = gazebo::event::Events::ConnectPreRender([this]() { Update(); });
  }

 private:
  void Update()
  {
    const auto laser = scene->GetVisual(laserName);
    if (!laser) return;
    // LaserVisual registers its update when created. Register clipping later,
    // so fresh scan vertices are clipped before the frame is drawn.
    if (!clipConnection)
      clipConnection = gazebo::event::Events::ConnectPreRender([this]() { Clip(); });
    auto node = laser->GetSceneNode();
    for (unsigned int i = 0; i < node->numAttachedObjects(); ++i)
    {
      auto line = dynamic_cast<gazebo::rendering::DynamicLines *>(node->getAttachedObject(i));
      if (!line) continue;
      const auto original = line->getMaterial();
      if (original->getName().find("Gazebo/") != 0) continue;
      const auto name = "SmartWheelChair/FaintLidar/" + laserName + "/" + original->getName();
      auto material = Ogre::MaterialManager::getSingleton().getByName(name);
      if (material.isNull())
      {
        material = original->clone(name);
        for (unsigned int t = 0; t < material->getNumTechniques(); ++t)
          for (unsigned int p = 0; p < material->getTechnique(t)->getNumPasses(); ++p)
          {
            auto pass = material->getTechnique(t)->getPass(p);
            pass->setDiffuse(color);
            pass->setSceneBlending(Ogre::SBT_TRANSPARENT_ALPHA);
            pass->setDepthWriteEnabled(false);
            // Gazebo's laser scripts override diffuse alpha with manual alpha.
            for (unsigned int u = 0; u < pass->getNumTextureUnitStates(); ++u)
            {
              pass->getTextureUnitState(u)->setColourOperationEx(
                  Ogre::LBX_SOURCE1, Ogre::LBS_MANUAL, Ogre::LBS_CURRENT, color);
              pass->getTextureUnitState(u)->setAlphaOperation(
                  Ogre::LBX_SOURCE1, Ogre::LBS_MANUAL, Ogre::LBS_CURRENT, color.a);
            }
          }
      }
      GZ_OGRE_SET_MATERIAL_BY_NAME(line, name);
    }
  }

  void Clip()
  {
    const auto laser = scene->GetVisual(laserName);
    if (!laser) return;
    const auto laserPose = laser->WorldPose();
    const auto origin = laserPose.Rot().RotateVectorReverse(
        housing->WorldPose().Pos() - laserPose.Pos());
    const auto toSensor = housing->WorldPose().Rot().Inverse() * laserPose.Rot();
    struct Box
    {
      ignition::math::Vector3d origin, half;
      ignition::math::Quaterniond rotation;
    };
    std::vector<Box> boxes;
    const auto body = housing->GetParent();
    for (unsigned int i = 0; i < body->GetChildCount(); ++i)
    {
      const auto visual = body->GetChild(i);
      // All current base-link body pieces are boxes. Exclude this laser's
      // enclosing housing; keep the opposite housing and structural pieces.
      if (visual == housing || visual->GetGeometryType() != "box") continue;
      const auto pose = visual->WorldPose();
      boxes.push_back({pose.Rot().RotateVectorReverse(
                           housing->WorldPose().Pos() - pose.Pos()),
                       visual->GetGeometrySize() * .5,
                       pose.Rot().Inverse() * laserPose.Rot()});
    }
    auto node = laser->GetSceneNode();
    for (unsigned int i = 0; i < node->numAttachedObjects(); ++i)
    {
      auto line = dynamic_cast<gazebo::rendering::DynamicLines *>(node->getAttachedObject(i));
      if (!line) continue;
      bool changed = false;
      for (unsigned int p = 0; p < line->GetPointCount(); ++p)
      {
        const auto delta = line->Point(p) - origin;
        const auto sensorDelta = toSensor * delta;
        const double angle = std::atan2(sensorDelta.Y(), sensorDelta.X());
        if (angle < lowerAngle-1e-5 || angle > upperAngle+1e-5)
        {
          line->SetPoint(p, origin);
          changed = true;
          continue;
        }
        double limit = 1.;
        for (const auto &box : boxes)
        {
          const auto direction = box.rotation * delta;
          double enter = 0., leave = limit;
          for (int axis = 0; axis < 3 && enter <= leave; ++axis)
          {
            if (std::abs(direction[axis]) < 1e-10)
            {
              if (std::abs(box.origin[axis]) > box.half[axis]) leave = -1.;
              continue;
            }
            double a = (-box.half[axis] - box.origin[axis]) / direction[axis];
            double b = (box.half[axis] - box.origin[axis]) / direction[axis];
            if (a > b) std::swap(a, b);
            enter = std::max(enter, a);
            leave = std::min(leave, b);
          }
          if (enter <= leave) limit = enter;
        }
        if (limit < 1. - 1e-8)
        {
          line->SetPoint(p, origin + delta * limit);
          changed = true;
        }
      }
      if (changed) line->Update();
    }
  }

  gazebo::rendering::ScenePtr scene;
  gazebo::rendering::VisualPtr housing;
  std::string laserName;
  Ogre::ColourValue color;
  double lowerAngle = 0., upperAngle = 0.;
  gazebo::event::ConnectionPtr connection;
  gazebo::event::ConnectionPtr clipConnection;
};

GZ_REGISTER_VISUAL_PLUGIN(FaintLidarVisualPlugin)
}  // namespace smart_wheelchair_gazebo
