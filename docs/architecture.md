# System Architecture

The supplied project report describes three main layers:

1. **Local hospital/client layer**
   - Chest X-ray data remains local.
   - Images are preprocessed locally.
   - Local training is performed.
   - Only model updates are sent to the server.

2. **Centralized Authorization Authority (CAA)**
   - Registers hospitals/devices.
   - Uses HMAC-SHA256-based authentication tokens.
   - Verifies participating clients before model updates are accepted.

3. **Federated aggregation layer**
   - Receives authenticated model updates.
   - Aggregates updates using Federated Averaging (FedAvg).
   - Redistributes the updated global model.

The report additionally describes OSDSTran for local medical-image classification and PDOA for hyperparameter tuning.
