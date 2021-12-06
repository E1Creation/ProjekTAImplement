firebase.performance();

// TODO: Create a custom trace to monitor image upload.
const trace = firebase.performance().trace('saveImageMessage');

// TODO: Start the "timer" for the custom trace.
trace.start();

// TODO: Record image size.
trace.putMetric('imageSize', file.size);

    // TODO: Stop the "timer" for the custom trace.
    trace.stop();