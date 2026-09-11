// Run via check_trajectory_preview.py under Xvfb, against a static test fixture.
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <gazebo/gazebo_client.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/rendering/DynamicLines.hh>
#include <gazebo/rendering/ogre_gazebo.h>
#include <geometry_msgs/PolygonStamped.h>
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
  Require(message->ranges.size() == 721, "lidar does not have 721 beams");
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
  const bool doorsOnly = argc > 1 && std::string(argv[1]) == "--doors-only";
  const bool bodyClearance = argc > 1 && std::string(argv[1]) == "--body-clearance";
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
    auto doors = node.advertise<geometry_msgs::PolygonStamped>("/shared_control/door_detections", 1);
    const auto step = [&]() {
      ros::spinOnce();
      scene->PreRender();
      gazebo::event::Events::preRender();
      ros::WallDuration(.01).sleep();
    };
    const auto deadline = ros::WallTime::now() + ros::WallDuration(30);
    while ((!scene->GetVisual("smart_wheelchair") ||
            raw.getNumSubscribers() == 0 || final.getNumSubscribers() == 0 ||
            doors.getNumSubscribers() == 0) &&
           ros::WallTime::now() < deadline) step();
    auto parent = scene->GetVisual("smart_wheelchair");
    std::cout << "Initialized: parent=" << bool(parent) << ", scans=" << samples[0] << "/" << samples[1]
              << ", command subscribers=" << raw.getNumSubscribers() << "/" << final.getNumSubscribers()
              << ", scene visuals=" << scene->VisualCount() << std::endl;
    Require(doors.getNumSubscribers() > 0, "door display subscriber did not initialize");
    if (!doorsOnly)
    {
      while ((samples[0] < 5 || samples[1] < 5) && ros::WallTime::now() < deadline) step();
      Require(parent && samples[0] >= 5 && samples[1] >= 5, "scene/lidars did not initialize");
    }
    Require(bool(parent), "scene did not initialize");
    if (!doorsOnly) for (const auto &side : {"left", "right"})
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
        const Ogre::ColourValue expected = std::string(side) == "left"
            ? Ogre::ColourValue(.1f, .65f, 1.f, .08f)
            : Ogre::ColourValue(.75f, .1f, 1.f, .08f);
        Require(pass->getDiffuse() == expected, "left/right lidar color incorrect");
        Require(std::abs(pass->getDiffuse().a - .08f) < 1e-6 && !pass->getDepthWriteEnabled(),
                "native lidar opacity/depth writing incorrect");
        Require(pass->getSourceBlendFactor() == Ogre::SBF_SOURCE_ALPHA &&
                pass->getDestBlendFactor() == Ogre::SBF_ONE_MINUS_SOURCE_ALPHA,
                "native lidar does not alpha blend");
        for (unsigned int u = 0; u < pass->getNumTextureUnitStates(); ++u)
        {
          const auto &alpha = pass->getTextureUnitState(u)->getAlphaBlendMode();
          Require(alpha.source1 == Ogre::LBS_MANUAL && std::abs(alpha.alphaArg1 - .08f) < 1e-6,
                  "texture unit overrides faint alpha");
          const auto &color = pass->getTextureUnitState(u)->getColourBlendMode();
          Require(color.source1 == Ogre::LBS_MANUAL && color.colourArg1 == expected,
                  "texture unit overrides left/right scan color");
        }
        Require(line->getVisibilityFlags() == GZ_VISIBILITY_GUI,
                "native lidar visible to camera sensors");
        if (line->GetPointCount() == 1442)
        {
          auto sensor = scene->GetVisual(std::string("smart_wheelchair::base_link::") +
                                         side + "_lidar_body_visual");
          const auto origin = sensor->WorldPose().Pos();
          if (bodyClearance)
          {
            if (line->GetOperationType() == gazebo::rendering::RENDERING_LINE_LIST)
            {
              const bool leftSide = std::string(side) == "left";
              const int first = leftSide ? 300 : 20;
              const int last = leftSide ? 700 : 420;
              unsigned int visible = 0;
              for (int b = 0; b <= 720; ++b)
              {
                const double length = (laser->WorldPose().CoordPositionAdd(
                    line->Point(2*b+1))-origin).Length();
                if (b >= first && b <= last)
                {
                  const double raw = scans[leftSide ? 0 : 1][b];
                  const double expected = std::isfinite(raw) ? std::max(.08, raw) : 5.;
                  Require(std::abs(length-expected) < .003,
                          "production body clips an in-sector scan ray");
                  ++visible;
                }
                else Require(length < .001, "out-of-sector ray remains visible");
              }
              Require(visible == 401, "selected sector must keep all 401 rays");
              std::cout << side << " production body: 401/401 selected rays unshortened" << std::endl;
            }
          }
          else
          {
          const unsigned int beam = std::string(side) == "left" ? 630 : 90;
          const auto end = laser->WorldPose().CoordPositionAdd(line->Point(2*beam+1));
          Require(std::abs((end - origin).Length() - (.3-.05*std::sqrt(2.))) < .002,
                  "GUI ray passes through visual-only body occluder");
          const auto rear = laser->WorldPose().CoordPositionAdd(line->Point(1));
          Require((rear-origin).Length() < .001,
                  "GUI rear ray remains outside selected 200 degree view");
          const auto front = laser->WorldPose().CoordPositionAdd(line->Point(721));
          std::cout << side << " occlusion: back=" << (end-origin).Length()
                    << ", front=" << (front-origin).Length() << std::endl;
          // The hit-only strip collapses infinite returns to range_min.
          Require((front - origin).Length() > .5 ||
                  std::abs((front - origin).Length() - .08) < .001,
                  "unblocked front ray was clipped");
          if (line->GetOperationType() == gazebo::rendering::RENDERING_LINE_LIST)
            Require((front - origin).Length() > .5,
                    "off-plane body visual incorrectly shadows front ray");
          Require(scans[std::string(side) == "left" ? 0 : 1][beam] > .5,
                  "visual-only body occluder changed raw ROS scan");
          }
        }
        ++faintObjects;
      }
      Require(faintObjects == 4, "missing native lidar strips/fan/ray lines");
      const auto original = Ogre::MaterialManager::getSingleton().getByName("Gazebo/BlueLaser");
      Require(std::abs(original->getTechnique(0)->getPass(0)->getTextureUnitState(0)
                          ->getAlphaBlendMode().alphaArg1 - .4f) < 1e-6,
              "global laser material was modified");
      std::cout << side << " native lidar: 4 GUI-only render objects, distinct blue, alpha=0.08, original unchanged" << std::endl;
    }
    if (!doorsOnly)
    {
      baseline = scans;
      compareScans = true;
    }
    const auto initialPose = parent->WorldPose();

    geometry_msgs::PolygonStamped doorMessage;
    doorMessage.header.frame_id = "odom";
    doorMessage.header.stamp = ros::Time::now();
    geometry_msgs::Point32 endpoint;
    endpoint.x = -5.2f;
    endpoint.y = 0.4f;
    doorMessage.polygon.points.push_back(endpoint);
    endpoint.x = -4.2f;
    endpoint.y = 1.4f;
    doorMessage.polygon.points.push_back(endpoint);
    const auto doorDeadline = ros::WallTime::now() + ros::WallDuration(5);
    gazebo::rendering::VisualPtr door;
    while (!(door = scene->GetVisual("smart_wheelchair_door_detection_0")) &&
           ros::WallTime::now() < doorDeadline)
    {
      doorMessage.header.stamp = ros::Time::now();
      doors.publish(doorMessage);
      step();
    }
    Require(door && door->GetVisible(), "valid door segment was not rendered");
    Require(door->GetParent() == parent, "door segment has wrong parent");
    Require(door->GetVisibilityFlags() == GZ_VISIBILITY_GUI,
            "door segment is visible to sensor renderers");
    const auto doorPose = door->WorldPose();
    std::cout << "Door world pose: " << doorPose << ", parent: " << parent->WorldPose() << std::endl;
    Require(doorPose.Pos().Equal({-4.7, .9, .12}, 1e-4),
            "door segment is not anchored at its odom/world midpoint");
    const auto doorAxis = doorPose.Rot().RotateVector({0, 0, 1});
    Require(doorAxis.Equal({std::sqrt(.5), std::sqrt(.5), 0}, 1e-4),
            "door cylinder does not join its world endpoints");
    Require(door->Scale().Equal({.03, .03, std::sqrt(2.0)}, 1e-4),
            "door cylinder diameter/length incorrect");
    const auto doorMaterial = Ogre::MaterialManager::getSingleton().getByName(door->GetMaterialName());
    Require(!doorMaterial.isNull(), "door segment material missing");
    const auto doorColor = doorMaterial->getTechnique(0)->getPass(0)->getDiffuse();
    Require(std::abs(doorColor.r) < 1e-6 && std::abs(doorColor.g - 1.0f) < 1e-6 &&
            std::abs(doorColor.b) < 1e-6, "door segment is not green");

    auto capped = doorMessage;
    capped.polygon.points.clear();
    for (int i = 0; i < 33; ++i)
    {
      endpoint.x = -5.5f + .02f * i;
      endpoint.y = -.5f;
      capped.polygon.points.push_back(endpoint);
      endpoint.y = -.3f;
      capped.polygon.points.push_back(endpoint);
    }
    const auto capDeadline = ros::WallTime::now() + ros::WallDuration(2);
    gazebo::rendering::VisualPtr lastDoor;
    while (!(lastDoor = scene->GetVisual("smart_wheelchair_door_detection_31")) &&
           ros::WallTime::now() < capDeadline)
    {
      capped.header.stamp = ros::Time::now();
      doors.publish(capped);
      step();
    }
    Require(lastDoor && lastDoor->GetVisible(), "32nd capped door segment was not rendered");
    Require(!scene->GetVisual("smart_wheelchair_door_detection_32"),
            "door display exceeded its 32-segment cap");

    const auto requireRejected = [&](geometry_msgs::PolygonStamped rejected, const std::string &label) {
      doorMessage.header.stamp = ros::Time::now();
      doors.publish(doorMessage);
      const auto refreshEnd = ros::WallTime::now() + ros::WallDuration(.05);
      while (ros::WallTime::now() < refreshEnd) step();
      doors.publish(rejected);
      const auto end = ros::WallTime::now() + ros::WallDuration(.05);
      while (ros::WallTime::now() < end) step();
      Require(door->GetVisible(), label + " input changed the active door display");
      Require(door->WorldPose().Pos().Equal(doorPose.Pos(), 1e-6),
              label + " input moved the active door display");
    };
    auto rejected = doorMessage;
    rejected.polygon.points.pop_back();
    rejected.polygon.points[0].x += 3.0f;
    requireRejected(rejected, "odd-point");
    rejected = doorMessage;
    rejected.header.frame_id = "map";
    rejected.polygon.points[0].x += 3.0f;
    rejected.polygon.points[1].x += 3.0f;
    requireRejected(rejected, "wrong-frame");
    rejected = doorMessage;
    rejected.polygon.points[0].x = std::numeric_limits<float>::quiet_NaN();
    requireRejected(rejected, "nonfinite");
    rejected = doorMessage;
    rejected.header.stamp = ros::Time::now() - ros::Duration(1.0);
    rejected.polygon.points[0].x += 3.0f;
    rejected.polygon.points[1].x += 3.0f;
    requireRejected(rejected, "stale-stamp");

    doorMessage.header.stamp = ros::Time::now();
    doorMessage.polygon.points.clear();
    doors.publish(doorMessage);
    const auto clearDeadline = ros::WallTime::now() + ros::WallDuration(2);
    while (door->GetVisible() && ros::WallTime::now() < clearDeadline) step();
    Require(!door->GetVisible(), "empty door message did not clear display");

    doorMessage.header.stamp = ros::Time::now();
    endpoint.x = -5.2f;
    endpoint.y = .4f;
    endpoint.z = 9.0f;
    doorMessage.polygon.points.push_back(endpoint);
    endpoint.x = -4.2f;
    endpoint.y = 1.4f;
    endpoint.z = -9.0f;
    doorMessage.polygon.points.push_back(endpoint);
    doors.publish(doorMessage);
    const auto reactivateDeadline = ros::WallTime::now() + ros::WallDuration(2);
    while (!door->GetVisible() && ros::WallTime::now() < reactivateDeadline) step();
    Require(door->GetVisible() && std::abs(door->WorldPose().Pos().Z() - .12) < 1e-6,
            "door display did not reactivate at fixed world z");
    const auto doorExpiry = ros::WallTime::now() + ros::WallDuration(2);
    while (door->GetVisible() && ros::WallTime::now() < doorExpiry) step();
    Require(!door->GetVisible(), "door segment did not expire after reception timeout");
    if (doorsOnly)
      std::cout << "Door display: anchored green GUI-only segment; rejection, clear and expiry verified" << std::endl;

    if (!doorsOnly)
    {
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
    }
    door.reset();
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
