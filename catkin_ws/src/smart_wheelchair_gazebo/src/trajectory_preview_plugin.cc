#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/transport/transport.hh>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Twist.h>
#include <ignition/math/Color.hh>
#include <ignition/math/Pose3.hh>
#include <ros/callback_queue.h>
#include <ros/ros.h>

namespace smart_wheelchair_gazebo
{
namespace
{
using Clock = std::chrono::steady_clock;

struct CommandPreview
{
  double linear{0.0};
  double angular{0.0};
  Clock::time_point lastCommand{};
  std::set<std::string> visuals;
};

struct DoorDisplay
{
  std::vector<ignition::math::Vector3d> endpoints;
  Clock::time_point lastReception{};
  ros::Time messageStamp;
  std::set<std::string> visuals;
};
}  // namespace

class TrajectoryPreviewPlugin : public gazebo::ModelPlugin
{
 public:
  ~TrajectoryPreviewPlugin() override
  {
    this->updateConnection.reset();
    this->callbackQueue.disable();
    this->rawSubscriber.shutdown();
    this->filteredSubscriber.shutdown();
    this->doorSubscriber.shutdown();
    this->HideVisuals(this->rawCommand.visuals);
    this->HideVisuals(this->filteredCommand.visuals);
    this->HideVisuals(this->doors.visuals);
  }

  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    if (!ros::isInitialized())
    {
      gzerr << "Trajectory preview requires gazebo_ros_api_plugin.\n";
      return;
    }
    this->model = model;
    this->parentName = model->GetScopedName();
    const auto read = [&sdf](const std::string &name, double &value)
    {
      if (sdf->HasElement(name))
        value = sdf->Get<double>(name);
    };
    read("prediction_seconds", this->predictionSeconds);
    read("step_seconds", this->stepSeconds);
    read("command_timeout_s", this->commandTimeout);
    read("rear_axle_x_m", this->rearAxleX);
    read("wheel_separation_m", this->wheelSeparation);
    read("z_m", this->z);
    if (!std::isfinite(this->predictionSeconds) || this->predictionSeconds <= 0.0 ||
        !std::isfinite(this->stepSeconds) || this->stepSeconds <= 0.0 ||
        !std::isfinite(this->commandTimeout) || this->commandTimeout <= 0.0 ||
        !std::isfinite(this->rearAxleX) || !std::isfinite(this->wheelSeparation) ||
        this->wheelSeparation <= 0.0 || !std::isfinite(this->z) ||
        this->predictionSeconds / this->stepSeconds > 1000.0)
    {
      gzerr << "Invalid trajectory preview parameters.\n";
      return;
    }
    std::string rawTopic = "/cmd_vel_raw";
    std::string filteredTopic = "/cmd_vel";
    if (sdf->HasElement("cmd_topic"))
      rawTopic = sdf->Get<std::string>("cmd_topic");
    if (sdf->HasElement("filtered_cmd_topic"))
      filteredTopic = sdf->Get<std::string>("filtered_cmd_topic");

    this->transport.reset(new gazebo::transport::Node);
    this->transport->Init(model->GetWorld()->Name());
    this->visualPublisher = this->transport->Advertise<gazebo::msgs::Visual>("~/visual", 1000);
    this->rosNode.reset(new ros::NodeHandle);
    this->rosNode->setCallbackQueue(&this->callbackQueue);
    this->rawSubscriber = this->rosNode->subscribe(rawTopic, 1, &TrajectoryPreviewPlugin::OnRaw, this);
    this->filteredSubscriber = this->rosNode->subscribe(filteredTopic, 1, &TrajectoryPreviewPlugin::OnFiltered, this);
    this->doorSubscriber = this->rosNode->subscribe(
        "/shared_control/door_detections", 1, &TrajectoryPreviewPlugin::OnDoors, this);
    this->updateConnection = gazebo::event::Events::ConnectWorldUpdateBegin(
        std::bind(&TrajectoryPreviewPlugin::Update, this));
  }

 private:
  void OnRaw(const geometry_msgs::Twist::ConstPtr &message)
  {
    this->SetCommand(this->rawCommand, *message);
  }

  void OnFiltered(const geometry_msgs::Twist::ConstPtr &message)
  {
    this->SetCommand(this->filteredCommand, *message);
  }

  void OnDoors(const geometry_msgs::PolygonStamped::ConstPtr &message)
  {
    const auto &points = message->polygon.points;
    const double stampAge = (ros::Time::now() - message->header.stamp).toSec();
    if (message->header.frame_id != "odom" || points.size() % 2 != 0 ||
        !std::isfinite(stampAge) || stampAge > this->doorTimeout)
      return;
    for (const auto &point : points)
      if (!std::isfinite(point.x) || !std::isfinite(point.y))
        return;

    this->doors.endpoints.clear();
    const size_t pointCount = std::min(points.size(), this->maxDoorSegments * 2);
    this->doors.endpoints.reserve(pointCount);
    for (size_t i = 0; i < pointCount; ++i)
      this->doors.endpoints.emplace_back(points[i].x, points[i].y, this->doorZ);
    this->doors.lastReception = Clock::now();
    this->doors.messageStamp = message->header.stamp;
  }

  void SetCommand(CommandPreview &state, const geometry_msgs::Twist &message)
  {
    const bool valid = std::isfinite(message.linear.x) && std::isfinite(message.angular.z);
    state.linear = valid ? message.linear.x : 0.0;
    state.angular = valid ? message.angular.z : 0.0;
    state.lastCommand = Clock::now();
  }

  void Update()
  {
    // All command callbacks and visual state changes run on the simulation thread.
    this->callbackQueue.callAvailable(ros::WallDuration(0));
    const auto now = Clock::now();
    if (std::chrono::duration<double>(now - this->lastRender).count() < 0.1)
      return;
    this->lastRender = now;
    this->Render(this->rawCommand, "smart_wheelchair_trajectory_raw", this->rawMarkerDiameter, 0.26);
    this->Render(this->filteredCommand, "smart_wheelchair_trajectory_filtered", this->filteredMarkerDiameter, 0.34);
    this->RenderDoors();
  }

  void RenderDoors()
  {
    const double receptionAge = std::chrono::duration<double>(Clock::now() - this->doors.lastReception).count();
    const double stampAge = (ros::Time::now() - this->doors.messageStamp).toSec();
    const bool active = receptionAge <= this->doorTimeout && std::isfinite(stampAge) &&
                        stampAge <= this->doorTimeout;
    std::set<std::string> nextVisuals;
    if (active)
    {
      const auto modelPose = this->model->WorldPose();
      for (size_t i = 0; i + 1 < this->doors.endpoints.size(); i += 2)
      {
        const auto worldDelta = this->doors.endpoints[i + 1] - this->doors.endpoints[i];
        if (worldDelta.Length() < 1e-9)
          continue;
        const std::string name = "smart_wheelchair_door_detection_" + std::to_string(i / 2);
        const auto worldMidpoint = (this->doors.endpoints[i] + this->doors.endpoints[i + 1]) / 2.0;
        const auto localMidpoint = modelPose.Rot().RotateVectorReverse(worldMidpoint - modelPose.Pos());
        const auto localDelta = modelPose.Rot().RotateVectorReverse(worldDelta);
        ignition::math::Quaterniond rotation;
        rotation.From2Axes({0, 0, 1}, localDelta);

        gazebo::msgs::Visual visual;
        visual.set_name(name);
        visual.set_parent_name(this->parentName);
        visual.set_is_static(false);
        visual.set_visible(true);
        visual.set_cast_shadows(false);
        auto guiOnly = visual.add_plugin();
        guiOnly->set_name("gui_only");
        guiOnly->set_filename("libSmartWheelChairGuiOnlyVisual.so");
        auto geometry = visual.mutable_geometry();
        geometry->set_type(gazebo::msgs::Geometry::CYLINDER);
        geometry->mutable_cylinder()->set_radius(this->doorDiameter / 2.0);
        geometry->mutable_cylinder()->set_length(worldDelta.Length());
        gazebo::msgs::Set(visual.mutable_pose(), ignition::math::Pose3d(localMidpoint, rotation));
        const ignition::math::Color green(0, 1, 0, 1);
        gazebo::msgs::Set(visual.mutable_material()->mutable_ambient(), green);
        gazebo::msgs::Set(visual.mutable_material()->mutable_diffuse(), green);
        gazebo::msgs::Set(visual.mutable_material()->mutable_emissive(), green);
        this->visualPublisher->Publish(visual);
        nextVisuals.insert(name);
      }
    }
    auto hiddenVisuals = this->doors.visuals;
    for (const auto &name : nextVisuals)
      hiddenVisuals.erase(name);
    this->HideVisuals(hiddenVisuals);
    this->doors.visuals.insert(nextVisuals.begin(), nextVisuals.end());
  }

  std::vector<ignition::math::Vector3d> Path(double linear, double angular, double yOffset) const
  {
    std::vector<ignition::math::Vector3d> points;
    const int steps = std::max(1, static_cast<int>(std::round(this->predictionSeconds / this->stepSeconds)));
    for (int i = 0; i <= steps; ++i)
    {
      const double t = std::min(i * this->stepSeconds, this->predictionSeconds);
      double baseX = linear * t;
      double baseY = 0.0;
      double heading = 0.0;
      if (std::abs(angular) >= 1e-6)
      {
        heading = angular * t;
        const double radius = linear / angular;
        baseX = radius * std::sin(heading);
        baseY = radius * (1.0 - std::cos(heading));
      }
      points.emplace_back(
          baseX + this->rearAxleX * std::cos(heading) - yOffset * std::sin(heading),
          baseY + this->rearAxleX * std::sin(heading) + yOffset * std::cos(heading), this->z);
    }
    return points;
  }

  void Render(CommandPreview &state, const std::string &prefix, double diameter, double lift)
  {
    const bool active = std::chrono::duration<double>(Clock::now() - state.lastCommand).count()
                            <= this->commandTimeout &&
                        (std::abs(state.linear) >= 1e-4 || std::abs(state.angular) >= 1e-4);
    std::set<std::string> nextVisuals;
    if (active)
    {
      const std::array<std::string, 3> wheels{{"center", "left", "right"}};
      const std::array<double, 3> offsets{{0.0, this->wheelSeparation / 2.0, -this->wheelSeparation / 2.0}};
      const std::array<ignition::math::Color, 3> colors{{{0, 1, 1, 1}, {1, 0, 0, 1}, {1, 1, 0, 1}}};
      for (size_t wheel = 0; wheel < wheels.size(); ++wheel)
      {
        const auto points = this->Path(state.linear, state.angular, offsets[wheel]);
        for (size_t i = 0; i + 1 < points.size(); ++i)
        {
          const auto delta = points[i + 1] - points[i];
          if (delta.Length() < 1e-9)
            continue;
          const std::string name = prefix + "_" + wheels[wheel] + "_" + std::to_string(i);
          gazebo::msgs::Visual visual;
          visual.set_name(name);
          visual.set_parent_name(this->parentName);
          visual.set_is_static(false);
          visual.set_visible(true);
          visual.set_cast_shadows(false);
          auto guiOnly = visual.add_plugin();
          guiOnly->set_name("gui_only");
          guiOnly->set_filename("libSmartWheelChairGuiOnlyVisual.so");
          auto geometry = visual.mutable_geometry();
          geometry->set_type(gazebo::msgs::Geometry::CYLINDER);
          geometry->mutable_cylinder()->set_radius(diameter / 2.0);
          geometry->mutable_cylinder()->set_length(delta.Length());
          ignition::math::Quaterniond rotation;
          rotation.From2Axes({0, 0, 1}, delta);
          const auto midpoint = (points[i] + points[i + 1]) / 2.0 + ignition::math::Vector3d(0, 0, lift);
          gazebo::msgs::Set(visual.mutable_pose(), ignition::math::Pose3d(midpoint, rotation));
          gazebo::msgs::Set(visual.mutable_material()->mutable_ambient(), colors[wheel]);
          gazebo::msgs::Set(visual.mutable_material()->mutable_diffuse(), colors[wheel]);
          gazebo::msgs::Set(visual.mutable_material()->mutable_emissive(), colors[wheel]);
          this->visualPublisher->Publish(visual);
          nextVisuals.insert(name);
        }
      }
    }
    auto hiddenVisuals = state.visuals;
    for (const auto &name : nextVisuals)
      hiddenVisuals.erase(name);
    this->HideVisuals(hiddenVisuals);
    // Retain known slots and keep publishing their desired hidden state, just
    // as active state is refreshed. A missed expiry packet must not leave a
    // stale path visible forever under rendering/transport load.
    state.visuals.insert(nextVisuals.begin(), nextVisuals.end());
  }

  // Gazebo retains deleted visuals in their parent's children. Keep bounded,
  // deterministic slots instead, and reuse the same render objects next time.
  void HideVisuals(const std::set<std::string> &names)
  {
    for (const auto &name : names)
    {
      gazebo::msgs::Visual visual;
      visual.set_name(name);
      visual.set_parent_name(this->parentName);
      visual.set_visible(false);
      this->visualPublisher->Publish(visual);
    }
  }

  ros::CallbackQueue callbackQueue;
  std::unique_ptr<ros::NodeHandle> rosNode;
  ros::Subscriber rawSubscriber, filteredSubscriber, doorSubscriber;
  gazebo::transport::NodePtr transport;
  gazebo::transport::PublisherPtr visualPublisher;
  gazebo::event::ConnectionPtr updateConnection;
  std::string parentName;
  gazebo::physics::ModelPtr model;
  CommandPreview rawCommand, filteredCommand;
  DoorDisplay doors;
  Clock::time_point lastRender{};
  double predictionSeconds{3.0};
  double stepSeconds{0.3};
  double commandTimeout{0.5};
  double rearAxleX{-0.33};
  double wheelSeparation{0.72};
  double z{0.18};
  double rawMarkerDiameter{0.03};
  double filteredMarkerDiameter{0.09};
  const double doorTimeout{0.6};
  const double doorDiameter{0.03};
  const double doorZ{0.12};
  const size_t maxDoorSegments{32};
};

GZ_REGISTER_MODEL_PLUGIN(TrajectoryPreviewPlugin)
}  // namespace smart_wheelchair_gazebo
