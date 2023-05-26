// Ros WEB SOCKET SERVER CONF

var ros = new ROSLIB.Ros({
  url : 'ws:192.168.67.94:9090'
});

ros.on('connection', function() {
  document.getElementById("status").innerHTML = "Connected";
});

ros.on('error', function(error) {
  document.getElementById("status").innerHTML = "Error";
});

ros.on('close', function() {
  document.getElementById("status").innerHTML = "Closed";
});

var arr = [];

function mapLoad(){
  var viewer2D = new ROS2D.Viewer({
    divID : 'map',
    width : 700,
    height : 700
  });

  var gridClient = new ROS2D.OccupancyGridClient({
    ros : ros,
    rootObject : viewer2D.scene,
    image: 'turtlebot.png',
    continuous: true
  });

  var robotMarker = new ROS2D.ArrowShape({
    size : 0.2,
    strokeSize : 0.2,
    pulse : false
  });
  
  var poseTopic = new ROSLIB.Topic({
    ros : ros,
    name : '/slam_out_pose',
    messageType:'geometry_msgs/PoseStamped',
  });
  
  /*
  var poseTopic = new ROSLIB.Topic({
    ros : ros,
    name : '/tf',
    messageType: '/tf2_msgs/TFMessage',
    //messageType:'tf/tfMessage'
  });
  */
 
  let degree = 0;
  var posx = 0;
  var posy = 0;
  poseTopic.subscribe(
  function(pose) {
    posx = pose.pose.position.x;
    posy = pose.pose.position.y;
    robotMarker.x = posx;
    robotMarker.y = -posy;

    let orientationQuerter=pose.pose.orientation
    var q0 = orientationQuerter.w;
    var q1 = orientationQuerter.x;
    var q2 = orientationQuerter.y;
    var q3 = orientationQuerter.z;
    degree=-Math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)) * 180.0 / Math.PI
    robotMarker.rotation = degree;
  });
  
 /*
  poseTopic.subscribe(
    function(pose) {
      if (pose.transforms[0].child_frame_id != "base_footprint"){return;}
      posx = pose.transforms[0].transform.translation.x;
      posy = pose.transforms[0].transform.translation.y;
      robotMarker.x = posx;
      robotMarker.y = -posy;
      
      let orientationQuerter=pose.transforms[0].transform.rotation;
      var q0 = orientationQuerter.w;
      var q1 = orientationQuerter.x;
      var q2 = orientationQuerter.y;
      var q3 = orientationQuerter.z;
      degree=-Math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)) * 180.0 / Math.PI
      robotMarker.rotation = degree;
  });
  */

  //rescuer marker 
  var distanceTopic = new ROSLIB.Topic({
    ros : ros,
    name : '/distance',
    messageType: '/std_msgs/String',
    //messageType:'tf/tfMessage'
  });
  var rescuerMarkers = [];
  distanceTopic.subscribe(
      function (distance) {
      let data = distance.data.split('/');
      let id = Number(data[0])
      let rad = (degree < 0 ? 180 + (180 + degree) : degree) * (Math.PI / 180);
      gridClient.rootObject.removeChild(rescuerMarkers[id]);

      rescuerMarkers[id] = new ROS2D.ArrowShape({
        size : 0.5,
        strokeSize : 0.02,
        pulse: false,
        fillColor: createjs.Graphics.getRGB(255,0,0, 0.9)
      });

      let x = posx + (Number(data[1])* 0.01) * Math.cos(rad);
      let y = posy - (Number(data[1])* 0.01) * Math.sin(rad);

      rescuerMarkers[id].x = x;
      rescuerMarkers[id].y = -y;
      rescuerMarkers[id].rotation = -90.0;

      gridClient.rootObject.addChild(rescuerMarkers[Number(data[0])]);
  });
  
  gridClient.rootObject.addChild(robotMarker);

  resize = 0.5
  gridClient.on('change', function(){
    viewer2D.scaleToDimensions(gridClient.currentGrid.width * resize, gridClient.currentGrid.height * resize);
    viewer2D.shift(gridClient.currentGrid.pose.position.x * resize, gridClient.currentGrid.pose.position.y * resize);
  });
}

// window.onload는 최종에 있는거 한번만 실행됨
window.addEventListener('onload', 
console.log("mapload"),
mapLoad()
)