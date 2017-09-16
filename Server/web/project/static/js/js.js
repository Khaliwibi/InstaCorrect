var app = angular.module('instacorrect', []);

app.controller('mainCtrl', function($scope, $http){
  // Normale state
  $scope.state = "idle";
  $scope.correct = function(){
    $scope.state = "loading";
    var data = JSON.stringify({'sentence': $scope.textModel});
    var headers = {'Content-Type': 'application/json; charset=utf-8'};
    $http.post('/api/is_correct', data=data, headers=headers).then(function(response){
      correct_prob = response.data['correct'];
      console.log('correct_prob', correct_prob)
      if (correct_prob > 0.5) {
        is_correct(correct_prob)
      } else {
        not_correct(correct_prob)
      }
    }).catch(function(error){
      console.log(error)
    });
  }
  function is_correct(probability){
    $scope.state = "correct";
    $scope.probability = probability;
  }

  function not_correct(probability){
    $scope.state = "incorrect";
    $scope.probability = probability;
  }


});
