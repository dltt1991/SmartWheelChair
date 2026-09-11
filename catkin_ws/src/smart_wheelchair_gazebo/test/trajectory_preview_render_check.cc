// Run via check_trajectory_preview.py under Xvfb, against a static test fixture.
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <gazebo/gazebo_client.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/rendering/DynamicLines.hh>
#include <gazebo/rendering/ogre_gazebo.h>
#include <geometry_msgs/Twist.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>

namespace
{
void Require(bool value, const std::string &message)
{
  if (!value) throw std::runtime_error(message);
}

std::array<std::vector<float>, 2> scans;
std::array<std::vector<float>, 2> baseline;
std::array<int, 2> samples{{0, 0}};
std::array<double, 2> firstStamp{{0, 0}}, lastStamp{{0, 0}};
bool compareScans = false;
bool scanChanged = false;
int changedBeams = 0;
int images = 0;

void Scan(const sensor_msgs::LaserScan::ConstPtr &message, size_t side)
{
  Require(message->ranges.size() == 401, "lidar does not have 401 beams");
  if (samples[side] == 0) firstStamp[side] = message->header.stamp.toSec();
  lastStamp[side] = message->header.stamp.toSec();
  scans[side] = message->ranges;
  ++samples[side];
  if (!compareScans) return;
  if (scans[side].size() != baseline[side].size()) { scanChanged = true; return; }
  for (size_t i = 0; i < scans[side].size(); ++i)
  {
    const float actual = scans[side][i], expected = baseline[side][i];
    if ((std::isfinite(actual) != std::isfinite(expected)) ||
        (std::isfinite(actual) && std::abs(actual - expected) > 0.001))
    {
      scanChanged = true;
      if (changedBeams++ < 12)
        std::cout << "Changed beam: side=" << side << ", angle="
                  << message->angle_min + i * message->angle_increment
                  << ", before=" << expected << ", after=" << actual << std::endl;
    }
  }
}
}  // namespace

int main(int argc, char **argv)
{
  ros::init(argc, argv, "trajectory_preview_render_check");
  gazebo::client::setup(argc, argv);
  gazebo::rendering::load();
  gazebo::rendering::init();
  auto scene = gazebo::rendering::create_scene("m6_room", true, false);
  int result = 0;
  try
  {
    ros::NodeHandle node;
    auto left = node.subscribe<sensor_msgs::LaserScan>("/scan_left", 100,
        +[](const sensor_msgs::LaserScan::ConstPtr &message) { Scan(message, 0); });
    auto right = node.subscribe<sensor_msgs::LaserScan>("/scan_right", 100,
        +[](const sensor_msgs::LaserScan::ConstPtr &message) { Scan(message, 1); });
    auto camera = node.subscribe<sensor_msgs::Image>("/camera/rear/image", 1,
        +[](const sensor_msgs::Image::ConstPtr &message) {
          Require(message->width > 0 && message->height > 0 &&
                  message->data.size() == message->height * message->step,
                  "invalid rear camera image");
          ++images;
        });
    auto raw = node.advertise<geometry_msgs::Twist>("/cmd_vel_raw", 1);
    auto final = node.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    const auto step = [&]() {
      ros::spinOnce();
      scene->PreRender();
      gazebo::event::Events::preRender();
      ros::WallDuration(.01).sleep();
    };
    const auto deadline = ros::WallTime::now() + ros::WallDuration(30);
    while ((!scene->GetVisual("smart_wheelchair") || samples[0] < 5 || samples[1] < 5 ||
            raw.getNumSubscribers() == 0 || final.getNumSubscribers() == 0) &&
           ros::WallTime::now() < deadline) step();
    auto parent = scene->GetVisual("smart_wheelchair");
    std::cout << "Initialized: parent=" << bool(parent) << ", scans=" << samples[0] << "/" << samples[1]
              << ", command subscribers=" << raw.getNumSubscribers() << "/" << final.getNumSubscribers()
              << ", scene visuals=" << scene->VisualCount() << std::endl;
    Require(parent && samples[0] >= 5 && samples[1] >= 5, "scene/lidars did not initialize");
    for (const auto &side : {"left", "right"})
    {
      const auto laserName = std::string("smart_wheelchair::base_link::") + side + "_lidar_GUIONLY_laser_vis";
      auto laser = scene->GetVisual(laserName);
      Require(laser && laser->GetVisible(), "lidar GUI visualization missing: " + laserName);
      const auto ready = ros::WallTime::now() + ros::WallDuration(5);
      while (laser->GetSceneNode()->numAttachedObjects() < 4 && ros::WallTime::now() < ready) step();
      step();  // Let the visual plugin style freshly allocated LaserVisual lines.
      unsigned int faintObjects = 0;
      auto laserNode = laser->GetSceneNode();
      for (unsigned int i = 0; i < laserNode->numAttachedObjects(); ++i)
      {
        auto line = dynamic_cast<gazebo::rendering::DynamicLines *>(laserNode->getAttachedObject(i));
        if (!line) continue;
        const auto material = line->getMaterial();
        Require(material->getName().find("SmartWheelChair/FaintLidar/") == 0,
                "native lidar still uses opaque default material");
        const auto pass = material->getTechnique(0)->getPass(0);
        Require(std::abs(pass->getDiffuse().a - .035f) < 1e-6 && !pass->getDepthWriteEnabled(),
                "native lidar opacity/depth writing incorrect");
        Require(pass->getSourceBlendFactor() == Ogre::SBF_SOURCE_ALPHA &&
                pass->getDestBlendFactor() == Ogre::SBF_ONE_MINUS_SOURCE_ALPHA,
                "native lidar does not alpha blend");
        for (unsigned int u = 0; u < pass->getNumTextureUnitStates(); ++u)
        {
          const auto &alpha = pass->getTextureUnitState(u)->getAlphaBlendMode();
          Require(alpha.source1 == Ogre::LBS_MANUAL && std::abs(alpha.alphaArg1 - .035f) < 1e-6,
                  "texture unit overrides faint alpha");
        }
        Require(line->getVisibilityFlags() == GZ_VISIBILITY_GUI,
                "native lidar visible to camera sensors");
        ++faintObjects;
      }
      Require(faintObjects == 4, "missing native lidar strips/fan/ray lines");
      const auto original = Ogre::MaterialManager::getSingleton().getByName("Gazebo/BlueLaser");
      Require(std::abs(original->getTechnique(0)->getPass(0)->getTextureUnitState(0)
                          ->getAlphaBlendMode().alphaArg1 - .4f) < 1e-6,
              "global laser material was modified");
      std::cout << side << " native lidar: 4 GUI-only render objects, alpha=0.035, original unchanged" << std::endl;
    }
    baseline = scans;
    compareScans = true;
    const auto initialPose = parent->WorldPose();

    std::vector<gazebo::rendering::VisualPtr> slots;
    const auto hold = [&](bool active, double seconds) {
      const auto end = ros::WallTime::now() + ros::WallDuration(seconds);
      geometry_msgs::Twist command;
      command.linear.x = .5;
      command.angular.z = .3;
      while (ros::WallTime::now() < end)
      {
        if (active)
        {
          raw.publish(command);
          if (argc < 2 || std::string(argv[1]) != "--raw-only") final.publish(command);
        }
        step();
        const auto pose = parent->WorldPose();
        Require(pose.Pos().Equal(initialPose.Pos(), 1e-7) &&
                pose.Rot().Equal(initialPose.Rot(), 1e-7), "model pose changed during comparison");
      }
    };
    if (argc > 1 && std::string(argv[1]) == "--baseline")
    {
      hold(false, 5);
      Require(!scanChanged, "baseline changed without preview commands");
      std::cout << "No-command baseline stable; valid camera frames=" << images << std::endl;
      scene.reset();
      gazebo::rendering::fini();
      gazebo::client::shutdown();
      return 0;
    }
    hold(true, 1.5);
    for (const auto &kind : {"raw", "filtered"})
    {
      if (argc > 1 && std::string(argv[1]) == "--raw-only" && std::string(kind) == "filtered") continue;
      for (const auto &wheel : {"center", "left", "right"})
        for (int i = 0; i < 10; ++i)
        {
          const auto name = std::string("smart_wheelchair_trajectory_") + kind + "_" + wheel + "_" + std::to_string(i);
          auto visual = scene->GetVisual(name);
          Require(visual && visual->GetVisible(), name + " not visible");
          Require(visual->GetParent() == parent, name + " has wrong parent");
          slots.push_back(visual);
        }
    }
    const auto childCount = parent->GetChildCount();
    const auto visualCount = scene->VisualCount();
    for (int cycle = 0; cycle < 3; ++cycle)
    {
      const auto expiryDeadline = ros::WallTime::now() + ros::WallDuration(5);
      while (ros::WallTime::now() < expiryDeadline)
      {
        step();
        bool hidden = true;
        for (const auto &visual : slots)
          hidden = hidden && !visual->GetVisible();
        if (hidden) break;
      }
      for (const auto &visual : slots)
      {
        Require(scene->GetVisual(visual->Name()) == visual, "expired slot was removed/replaced");
        Require(!visual->GetVisible(), visual->Name() + " expired slot remained visible");
      }
      hold(true, .8);
      for (const auto &visual : slots)
      {
        Require(scene->GetVisual(visual->Name()) == visual, "reactivated slot identity changed");
        Require(visual->GetVisible(), "reactivated slot stayed hidden");
      }
      Require(parent->GetChildCount() == childCount, "parent child count grew");
      Require(scene->VisualCount() == visualCount, "scene visual/name count grew");
    }
    std::cout << slots.size() << " stable slots: visible -> hidden -> visible, 3 cycles; children="
              << childCount << ", scene visuals=" << visualCount << std::endl;
    std::cout << "Scan comparison: changed=" << scanChanged << ", samples="
              << samples[0] << "/" << samples[1] << std::endl;
    for (const auto &visual : slots)
      Require(visual->GetVisibilityFlags() == GZ_VISIBILITY_GUI, "preview is visible to sensor renderers");
    Require(samples[0] > 10 && samples[1] > 10, "insufficient scan samples");
    Require(!scanChanged, "preview introduced lidar returns (tolerance 1 mm)");
    Require(images > 10, "insufficient camera frames");
    for (size_t side = 0; side < 2; ++side)
    {
      const double hz = (samples[side] - 1) / (lastStamp[side] - firstStamp[side]);
      Require(hz >= 9.5 && hz <= 10.5, "lidar did not maintain 10 Hz simulated rate");
      std::cout << "Lidar " << side << " rate=" << hz << " Hz (simulation time)" << std::endl;
    }
    std::cout << "Lidar unchanged across " << samples[0] << " left / " << samples[1]
              << " right scans; all GUI-only masks verified; valid camera frames=" << images << std::endl;
    hold(false, .9);
    slots.clear();
    parent.reset();
  }
  catch (const std::exception &error)
  {
    std::cerr << "FAIL: " << error.what() << std::endl;
    result = 1;
  }
  scene.reset();
  gazebo::rendering::fini();
  gazebo::client::shutdown();
  return result;
}
