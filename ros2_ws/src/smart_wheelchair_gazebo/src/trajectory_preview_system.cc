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

    this->transport.Subscribe(this->cmdTopic, &TrajectoryPreviewSystem::OnCmd, this);
    gzmsg << "SmartWheelChair trajectory preview subscribed to "
          << this->cmdTopic << std::endl;
  }

  void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;

    double linear = 0.0;
    double angular = 0.0;
    std::chrono::steady_clock::time_point stamp;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      linear = this->linear;
      angular = this->angular;
      stamp = this->lastCmdTime;
    }

    const auto age = std::chrono::steady_clock::now() - stamp;
    const bool active = (std::chrono::duration<double>(age).count() <= this->commandTimeout) &&
                        (std::abs(linear) >= 1e-4 || std::abs(angular) >= 1e-4);
    if (!active)
    {
      if (this->visible)
      {
        this->DeleteMarkers();
        this->visible = false;
        this->markerRendered = false;
      }
      return;
    }

    const bool commandChanged =
        std::abs(linear - this->lastRenderedLinear) >= 0.02 ||
        std::abs(angular - this->lastRenderedAngular) >= 0.02;
    const bool markerDue =
        !this->markerRendered ||
        std::chrono::duration<double>(_info.simTime - this->lastMarkerSimTime).count() >=
            this->markerPeriod;
    if (this->visible && !commandChanged && !markerDue)
    {
      return;
    }

    this->UpdateMarkers(_ecm, linear, angular);
    this->lastRenderedLinear = linear;
    this->lastRenderedAngular = angular;
    this->lastMarkerSimTime = _info.simTime;
    this->markerRendered = true;
    this->visible = true;
  }

 private:
  void OnCmd(const gz::msgs::Twist &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->linear = _msg.linear().x();
    this->angular = _msg.angular().z();
    this->lastCmdTime = std::chrono::steady_clock::now();
    if (!this->loggedCommand && (std::abs(this->linear) > 1e-4 || std::abs(this->angular) > 1e-4))
    {
      gzmsg << "SmartWheelChair trajectory preview received command"
            << std::endl;
      this->loggedCommand = true;
    }
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

  void UpdateMarkers(const gz::sim::EntityComponentManager &_ecm, double linear, double angular)
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
            points[static_cast<size_t>(i)],
            points[static_cast<size_t>(i + 1)],
            spec.color);
        this->transport.Request("/marker", marker);
      }
    }
  }

  void DeleteMarkers()
  {
    for (uint64_t id = 1; id <= static_cast<uint64_t>(this->maxSegments * 3); ++id)
    {
      gz::msgs::Marker marker;
      marker.set_action(gz::msgs::Marker::DELETE_MARKER);
      marker.set_ns("smart_wheelchair_trajectory");
      marker.set_id(id);
      this->transport.Request("/marker", marker);
    }
  }

  void AddCylinderMarker(
      gz::msgs::Marker &marker,
      uint64_t id,
      const gz::math::Vector3d &a,
      const gz::math::Vector3d &b,
      const gz::math::Color &color)
  {
    const auto delta = b - a;
    const double length = std::max(0.001, delta.Length());
    gz::math::Quaterniond rotation;
    rotation.SetFrom2Axes({0.0, 0.0, 1.0}, delta);

    marker.set_action(gz::msgs::Marker::ADD_MODIFY);
    marker.set_ns("smart_wheelchair_trajectory");
    marker.set_id(id);
    marker.set_type(gz::msgs::Marker::CYLINDER);
    marker.set_visibility(gz::msgs::Marker::ALL);
    marker.mutable_scale()->set_x(this->markerDiameter);
    marker.mutable_scale()->set_y(this->markerDiameter);
    marker.mutable_scale()->set_z(length);
    auto markerPose = marker.mutable_pose();
    markerPose->mutable_position()->set_x((a.X() + b.X()) / 2.0);
    markerPose->mutable_position()->set_y((a.Y() + b.Y()) / 2.0);
    markerPose->mutable_position()->set_z((a.Z() + b.Z()) / 2.0 + this->markerLift);
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
  double linear{0.0};
  double angular{0.0};
  double lastRenderedLinear{0.0};
  double lastRenderedAngular{0.0};
  double predictionSeconds{3.0};
  double stepSeconds{0.3};
  double commandTimeout{0.5};
  double markerPeriod{0.1};
  double markerDiameter{0.09};
  double markerLift{0.30};
  double rearAxleX{-0.33};
  double wheelSeparation{0.72};
  double z{0.18};
  int maxSegments{10};
  bool visible{false};
  bool loggedCommand{false};
  bool markerRendered{false};
  std::chrono::steady_clock::duration lastMarkerSimTime{0};
  std::chrono::steady_clock::time_point lastCmdTime{std::chrono::steady_clock::now()};
};
}  // namespace smart_wheelchair_gazebo

GZ_ADD_PLUGIN(
    smart_wheelchair_gazebo::TrajectoryPreviewSystem,
    gz::sim::System,
    smart_wheelchair_gazebo::TrajectoryPreviewSystem::ISystemConfigure,
    smart_wheelchair_gazebo::TrajectoryPreviewSystem::ISystemPreUpdate)
