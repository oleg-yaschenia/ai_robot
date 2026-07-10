#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "builtin_interfaces/msg/time.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace robot_camera_components
{

class ImageRateGateNode final : public rclcpp::Node
{
public:
  explicit ImageRateGateNode(const rclcpp::NodeOptions & options)
  : Node("image_rate_gate", options)
  {
    rate_hz_ = std::max(0.1, declare_parameter<double>("rate_hz", 4.5));
    diagnostics_period_sec_ = std::max(
      1.0,
      declare_parameter<double>("diagnostics_period_sec", 5.0));

    const auto publish_period = std::chrono::duration<double>(1.0 / rate_hz_);
    const auto publish_period_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(publish_period);

    auto input_qos = rclcpp::SensorDataQoS().keep_last(1);
    auto output_qos = rclcpp::QoS(rclcpp::KeepLast(1))
      .reliable()
      .durability_volatile();

    publisher_ = create_publisher<sensor_msgs::msg::Image>(
      "image_out",
      output_qos);

    subscription_ = create_subscription<sensor_msgs::msg::Image>(
      "image_in",
      input_qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
        on_image(std::move(message));
      });

    publish_timer_ = create_wall_timer(
      publish_period_ns,
      [this]() {
        publish_latest();
      });

    diagnostics_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(diagnostics_period_sec_)),
      [this]() {
        publish_diagnostics();
      });

    window_started_at_ = std::chrono::steady_clock::now();

    RCLCPP_INFO(
      get_logger(),
      "Image rate gate started: rate=%.2f Hz, input_qos=BEST_EFFORT/depth1, output_qos=RELIABLE/depth1, mode=latest-frame-timer",
      rate_hz_);
  }

private:
  static std::uint64_t stamp_to_ns(const builtin_interfaces::msg::Time & stamp)
  {
    return
      static_cast<std::uint64_t>(stamp.sec) * 1000000000ULL +
      static_cast<std::uint64_t>(stamp.nanosec);
  }

  void on_image(sensor_msgs::msg::Image::ConstSharedPtr message)
  {
    {
      std::lock_guard<std::mutex> lock(message_mutex_);
      latest_message_ = std::move(message);
    }
    input_count_.fetch_add(1, std::memory_order_relaxed);
  }

  void publish_latest()
  {
    sensor_msgs::msg::Image::ConstSharedPtr message;
    std::uint64_t stamp_ns = 0;

    {
      std::lock_guard<std::mutex> lock(message_mutex_);
      message = latest_message_;
      if (!message) {
        no_frame_count_.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      stamp_ns = stamp_to_ns(message->header.stamp);
      if (has_last_published_stamp_ && stamp_ns == last_published_stamp_ns_) {
        no_new_frame_count_.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      last_published_stamp_ns_ = stamp_ns;
      has_last_published_stamp_ = true;
    }

    // The rectified input reaches this component intra-process. Only the
    // latest selected frame is copied/serialized for the external Python
    // YOLO process.
    publisher_->publish(*message);
    output_count_.fetch_add(1, std::memory_order_relaxed);
  }

  void publish_diagnostics()
  {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
      std::chrono::duration<double>(now - window_started_at_).count();

    if (elapsed <= 0.0) {
      return;
    }

    const auto input = input_count_.exchange(0, std::memory_order_relaxed);
    const auto output = output_count_.exchange(0, std::memory_order_relaxed);
    const auto no_new =
      no_new_frame_count_.exchange(0, std::memory_order_relaxed);
    const auto no_frame =
      no_frame_count_.exchange(0, std::memory_order_relaxed);

    const double input_hz = static_cast<double>(input) / elapsed;
    const double output_hz = static_cast<double>(output) / elapsed;
    const std::uint64_t skipped = input > output ? input - output : 0;

    RCLCPP_INFO(
      get_logger(),
      "image_gate input=%.2fHz output=%.2fHz skipped=%llu no_new=%llu no_frame=%llu",
      input_hz,
      output_hz,
      static_cast<unsigned long long>(skipped),
      static_cast<unsigned long long>(no_new),
      static_cast<unsigned long long>(no_frame));

    window_started_at_ = now;
  }

  double rate_hz_{4.5};
  double diagnostics_period_sec_{5.0};

  std::mutex message_mutex_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_message_;
  std::uint64_t last_published_stamp_ns_{0};
  bool has_last_published_stamp_{false};

  std::atomic<std::uint64_t> input_count_{0};
  std::atomic<std::uint64_t> output_count_{0};
  std::atomic<std::uint64_t> no_new_frame_count_{0};
  std::atomic<std::uint64_t> no_frame_count_{0};
  std::chrono::steady_clock::time_point window_started_at_{};

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;
};

}  // namespace robot_camera_components

RCLCPP_COMPONENTS_REGISTER_NODE(
  robot_camera_components::ImageRateGateNode)
