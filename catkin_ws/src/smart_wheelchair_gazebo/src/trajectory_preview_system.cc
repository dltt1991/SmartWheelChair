#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include <gz/math/Color.hh>
#include <gz/math/Pose3.hh>
#include <gz/common/Console.hh>
#include <gz/msgs/marker.pb.h>
#include <gz/msgs/twist.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/transport/Node.hh>

namespace smart_wheelchair_gazebo
{
namespace
{
struct Point
{
  double x{0.0};
  double y{0.0};
  double z{0.18};
};

struct PathSpec
{
  double yOffset;
  gz::math::Color color;
};

struct CommandPreview
{
  double linear{0.0};
  double angular{0.0};
  double lastRenderedLinear{0.0};
  double lastRenderedAngular{0.0};
  bool visible{false};
  bool markerRendered{false};
  bool loggedCommand{false};
  std::chrono::steady_clock::duration lastMarkerSimTime{0};
  std::chrono::steady_clock::time_point lastCmdTime{std::chrono::steady_clock::now()};
};
}  // namespace

class TrajectoryPreviewSystem
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate
{
 public:
  void Configure(
      const gz::sim::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &,
      gz::sim::EventManager &) override
  {
    this->modelEntity = _entity;
    if (_sdf->HasElement("cmd_topic"))
      this->cmdTopic = _sdf->Get<std::string>("cmd_topic");
    if (_sdf->HasElement("filtered_cmd_topic"))
      this->filteredCmdTopic = _sdf->Get<std::string>("filtered_cmd_topic");
    if (_sdf->HasElement("prediction_seconds"))
      this->predictionSeconds = _sdf->Get<double>("prediction_seconds");
    if (_sdf->HasElement("step_seconds"))
      this->stepSeconds = _sdf->Get<double>("step_seconds");
    if (_sdf->HasElement("command_timeout_s"))
      this->commandTimeout = _sdf->Get<double>("command_timeout_s");
    if (_sdf->HasElement("rear_axle_x_m"))
      this->rearAxleX = _sdf->Get<double>("rear_axle_x_m");
    if (_sdf->HasElement("wheel_separation_m"))
      this->wheelSeparation = _sdf->Get<double>("wheel_separation_m");
    if (_sdf->HasElement("z_m"))
      this->z = _sdf->Get<double>("z_m");

    this->transport.Subscribe(this->cmdTopic, &TrajectoryPreviewSystem::OnRawCmd, this);
    this->transport.Subscribe(this->filteredCmdTopic, &TrajectoryPreviewSystem::OnFilteredCmd, this);
    gzmsg << "SmartWheelChair trajectory preview subscribed to "
          << this->cmdTopic << " and " << this->filteredCmdTopic << std::endl;
  }

  void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;

    std::lock_guard<std::mutex> lock(this->mutex);
    this->RenderPreview(_ecm, _info, this->rawCommand, "smart_wheelchair_trajectory_raw",
                        this->rawMarkerDiameter, this->rawMarkerLift);
    this->RenderPreview(_ecm, _info, this->filteredCommand, "smart_wheelchair_trajectory_filtered",
                        this->filteredMarkerDiameter, this->filteredMarkerLift);
  }

 private:
  void OnRawCmd(const gz::msgs::Twist &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->SetCommand(this->rawCommand, _msg, "raw");
  }

  void OnFilteredCmd(const gz::msgs::Twist &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->SetCommand(this->filteredCommand, _msg, "filtered");
  }

  void SetCommand(CommandPreview &command, const gz::msgs::Twist &_msg, const char *name)
  {
    command.linear = _msg.linear().x();
    command.angular = _msg.angular().z();
    command.lastCmdTime = std::chrono::steady_clock::now();
    if (!command.loggedCommand && (std::abs(command.linear) > 1e-4 || std::abs(command.angular) > 1e-4))
    {
      gzmsg << "SmartWheelChair trajectory preview received "
            << name << " command" << std::endl;
      command.loggedCommand = true;
    }
  }

  void RenderPreview(
      const gz::sim::EntityComponentManager &_ecm,
      const gz::sim::UpdateInfo &_info,
      CommandPreview &state,
      const std::string &markerNamespace,
      double diameter,
      double lift)
  {
    const auto age = std::chrono::steady_clock::now() - state.lastCmdTime;
    const bool active = (std::chrono::duration<double>(age).count() <= this->commandTimeout) &&
                        (std::abs(state.linear) >= 1e-4 || std::abs(state.angular) >= 1e-4);

    if (!active)
    {
      if (state.visible)
      {
        this->DeleteMarkers(markerNamespace);
        state.visible = false;
        state.markerRendered = false;
      }
      return;
    }

    const bool commandChanged =
        std::abs(state.linear - state.lastRenderedLinear) >= 0.02 ||
        std::abs(state.angular - state.lastRenderedAngular) >= 0.02;
    const bool markerDue =
        !state.markerRendered ||
        std::chrono::duration<double>(_info.simTime - state.lastMarkerSimTime).count() >=
            this->markerPeriod;
    if (state.visible && !commandChanged && !markerDue)
      return;

    this->UpdateMarkers(_ecm, state.linear, state.angular, markerNamespace, diameter, lift);
    state.lastRenderedLinear = state.linear;
    state.lastRenderedAngular = state.angular;
    state.lastMarkerSimTime = _info.simTime;
    state.markerRendered = true;
    state.visible = true;
  }

  std::vector<Point> Path(double linear, double angular, double yOffset) const
  {
    std::vector<Point> points;
    const int steps = std::max(1, static_cast<int>(std::round(this->predictionSeconds / this->stepSeconds)));
    points.reserve(static_cast<size_t>(steps + 1));
    for (int i = 0; i <= steps; ++i)
    {
      const double t = std::min(i * this->stepSeconds, this->predictionSeconds);
      double baseX = 0.0;
      double baseY = 0.0;
      double heading = 0.0;
      if (std::abs(angular) < 1e-6)
      {
        baseX = linear * t;
      }
      else
      {
        heading = angular * t;
        const double radius = linear / angular;
        baseX = radius * std::sin(heading);
        baseY = radius * (1.0 - std::cos(heading));
      }

      points.push_back({
          baseX + this->rearAxleX * std::cos(heading) - yOffset * std::sin(heading),
          baseY + this->rearAxleX * std::sin(heading) + yOffset * std::cos(heading),
          this->z});
    }
    return points;
  }

  void UpdateMarkers(
      const gz::sim::EntityComponentManager &_ecm,
      double linear,
      double angular,
      const std::string &markerNamespace,
      double diameter,
      double lift)
  {
    const std::array<PathSpec, 3> specs{{
        {0.0, {0.0f, 1.0f, 1.0f, 1.0f}},
        {this->wheelSeparation / 2.0, {1.0f, 0.0f, 0.0f, 1.0f}},
        {-this->wheelSeparation / 2.0, {1.0f, 1.0f, 0.0f, 1.0f}},
    }};
    const auto pose = gz::sim::worldPose(this->modelEntity, _ecm);

    for (size_t pathIndex = 0; pathIndex < specs.size(); ++pathIndex)
    {
      const auto &spec = specs[pathIndex];
      std::vector<gz::math::Vector3d> points;

      for (const auto &point : this->Path(linear, angular, spec.yOffset))
      {
        points.push_back(pose.Pos() + pose.Rot().RotateVector({point.x, point.y, point.z}));
      }

      const int count = std::min(this->maxSegments, static_cast<int>(points.size()) - 1);
      for (int i = 0; i < count; ++i)
      {
        gz::msgs::Marker marker;
        this->AddCylinderMarker(
            marker,
            static_cast<uint64_t>(pathIndex * this->maxSegments + i + 1),
            markerNamespace,
            points[static_cast<size_t>(i)],
            points[static_cast<size_t>(i + 1)],
            spec.color,
            diameter,
            lift);
        this->transport.Request("/marker", marker);
      }
    }
  }

  void DeleteMarkers(const std::string &markerNamespace)
  {
    for (uint64_t id = 1; id <= static_cast<uint64_t>(this->maxSegments * 3); ++id)
    {
      gz::msgs::Marker marker;
      marker.set_action(gz::msgs::Marker::DELETE_MARKER);
      marker.set_ns(markerNamespace);
      marker.set_id(id);
      this->transport.Request("/marker", marker);
    }
  }

  void AddCylinderMarker(
      gz::msgs::Marker &marker,
      uint64_t id,
      const std::string &markerNamespace,
      const gz::math::Vector3d &a,
      const gz::math::Vector3d &b,
      const gz::math::Color &color,
      double diameter,
      double lift)
  {
    const auto delta = b - a;
    const double length = std::max(0.001, delta.Length());
    gz::math::Quaterniond rotation;
    rotation.SetFrom2Axes({0.0, 0.0, 1.0}, delta);

    marker.set_action(gz::msgs::Marker::ADD_MODIFY);
    marker.set_ns(markerNamespace);
    marker.set_id(id);
    marker.set_type(gz::msgs::Marker::CYLINDER);
    marker.set_visibility(gz::msgs::Marker::ALL);
    marker.mutable_scale()->set_x(diameter);
    marker.mutable_scale()->set_y(diameter);
    marker.mutable_scale()->set_z(length);
    auto markerPose = marker.mutable_pose();
    markerPose->mutable_position()->set_x((a.X() + b.X()) / 2.0);
    markerPose->mutable_position()->set_y((a.Y() + b.Y()) / 2.0);
    markerPose->mutable_position()->set_z((a.Z() + b.Z()) / 2.0 + lift);
    markerPose->mutable_orientation()->set_x(rotation.X());
    markerPose->mutable_orientation()->set_y(rotation.Y());
    markerPose->mutable_orientation()->set_z(rotation.Z());
    markerPose->mutable_orientation()->set_w(rotation.W());
    this->SetMarkerMaterial(marker, color);
  }

  void SetMarkerMaterial(gz::msgs::Marker &marker, const gz::math::Color &color)
  {
    auto material = marker.mutable_material();
    material->mutable_ambient()->set_r(color.R());
    material->mutable_ambient()->set_g(color.G());
    material->mutable_ambient()->set_b(color.B());
    material->mutable_ambient()->set_a(color.A());
    material->mutable_diffuse()->set_r(color.R());
    material->mutable_diffuse()->set_g(color.G());
    material->mutable_diffuse()->set_b(color.B());
    material->mutable_diffuse()->set_a(color.A());
    material->mutable_emissive()->set_r(color.R());
    material->mutable_emissive()->set_g(color.G());
    material->mutable_emissive()->set_b(color.B());
    material->mutable_emissive()->set_a(color.A());
  }

  gz::transport::Node transport;
  std::mutex mutex;
  gz::sim::Entity modelEntity{gz::sim::kNullEntity};
  std::string cmdTopic{"/cmd_vel_raw"};
  std::string filteredCmdTopic{"/cmd_vel"};
  CommandPreview rawCommand;
  CommandPreview filteredCommand;
  double predictionSeconds{3.0};
  double stepSeconds{0.3};
  double commandTimeout{0.5};
  double markerPeriod{0.1};
  double rawMarkerDiameter{0.03};
  double filteredMarkerDiameter{0.09};
  double rawMarkerLift{0.26};
  double filteredMarkerLift{0.34};
  double rearAxleX{-0.33};
  double wheelSeparation{0.72};
  double z{0.18};
  int maxSegments{10};
};
}  // namespace smart_wheelchair_gazebo

GZ_ADD_PLUGIN(
    smart_wheelchair_gazebo::TrajectoryPreviewSystem,
    gz::sim::System,
    smart_wheelchair_gazebo::TrajectoryPreviewSystem::ISystemConfigure,
    smart_wheelchair_gazebo::TrajectoryPreviewSystem::ISystemPreUpdate)
