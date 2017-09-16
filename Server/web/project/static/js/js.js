var app = angular.module('instacorrect', []);

app.controller('mainCtrl', function($scope, $http){
  // Normale state
  $scope.state = "idle";
  $scope.correct = function(){
    $scope.state = "loading";
    var data = {'sentence': $scope.textModel};
    var headers = {'Content-Type': 'application/json; charset=utf-8'};
    $http.post('/api/is_correct', data=data, headers=headers).then(function(response){
      console.log(response)
    });
  }
  function is_correct(probability){
    $scope.state = "correct";
    $scope.probability = 0.98;
  }

  function not_correct(probability){
    $scope.state = "incorrect";
    $scope.probability = 0.98;
  }


});
