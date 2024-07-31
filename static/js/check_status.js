
        function check_status() {
            $.ajax({
                url: "{{ url_for('task_status', task_id=task_id) }}",
                success: function(data) {
                    if (data.status === "SUCCESS") {
                        window.location.href = data.result;
                    } else if (data.status === "PENDING") {
                        $("#status-message").text("Still processing...");
                        setTimeout(check_status, 2000); // Retry after 2 seconds
                    } else {
                        $("#status-message").text("An error occurred.");
                    }
                }
            });
        }

        $(document).ready(function() {
            check_status();
        });