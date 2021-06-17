import rclpy
from rclpy.node import Node
from rclpy.task import Future
from std_srvs.srv import Empty





reset_request = Empty.Request()
pause_request = Empty.Request()
unpause_request = Empty.Request()


#function needed to be called to reset the world.
class resetWorld(Node):
     def __init__(self):
        super().__init__('resetGazeboEnvironment')

        #creating a node
    # rclpy.init()
    # node = rclpy.create_node('reset_gazebo_world')
    # node.get_logger().info('Created node')
        reset_gazebo_world= self.create_client(Empty, '/reset_world')
        while not reset_gazebo_world.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        resp = reset_gazebo_world.call_async(reset_request)
        rclpy.spin_until_future_complete(self,resp)



def main2(args=None):
    
    #rclpy.init(args=args)

    restWorld = resetWorld()
    rclpy.spin(restWorld)

    restWorld.destroy_node()
    rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# def pauseWorld():
#         #creating a node
#     rclpy.init(args=sys.argv)
#     node = rclpy.create_node('pause_gazebo_world')
#     node.get_logger().info('Created node')    
#     pause_gazebo_world= node.create_client(Empty, '/pause_physics')
#     while not pause_gazebo_world.wait_for_service(timeout_sec=1.0):
#         node.get_logger().info('service not available, waiting again...')
#     resp = pause_gazebo_world.call_async(pause_request)
#     rclpy.spin_until_future_complete(node,resp)

# def unpauseWorld():
#         #creating a node
#     rclpy.init(args=sys.argv)
#     node = rclpy.create_node('unpause_gazebo_world')
#     node.get_logger().info('Created node')    
    unpause_gazebo_world= node.create_client(Empty, '/unpause_physics')
    while not unpause_gazebo_world.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')
    resp = unpause_gazebo_world.call_async(unpause_request)
    rclpy.spin_until_future_complete(node,resp)

# # t1=time.time()
# # unpauseWorld()
# # t2=time.time()
# # print(t2-t1)