function CheckStatus(taskId, taskStatusUrl) {
    $.ajax({
        url: taskStatusUrl + taskId,
        success: function(data) {
            if (data.status === "SUCCESS") {
                $("#loading-message").hide();
                $("#loader").hide();
                $("#plot-frame").attr("src", data.result).show();
                $("#btn").show();
                $("#info").show();
            } else if (data.status === "PENDING") {
                setTimeout(function() { CheckStatus(taskId, taskStatusUrl); }, 2000); // Retry after 2 seconds
                $("#btn").hide();
                $("#info").hide();
            } else {
                $("#loading-message").text("An error occurred.");
                $("#btn").hide();
                $("#info").hide();
                $("#loader").hide();
            }
        },
        error: function(xhr, status, error) {
            console.error("An error occurred: " + status + " " + error);
            $("#loading-message").text("An error occurred while checking the status.");
            $("#loader").hide();
            $("#btn").hide();
            $("#info").hide();
        }
    });
}

$(document).ready(function() {
    CheckStatus(taskId, taskStatusUrl);
});