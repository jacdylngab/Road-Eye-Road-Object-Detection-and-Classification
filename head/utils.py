
        # Set defaults for optional configs
        if loss_cls is None:
            loss_cls = { # Classification Loss that answers what the object is
                "type": "FocalLoss", # Downweight easy examples (background) and focus on hard ones
                "use_sigmoid": True, # used for multi-label classification
                "gamma": 2.0, # How hard we downweight easy examples
                "alpha": 0.25, # Balance positives vs negatives
                "loss_weight": 1.0 # It is a multiplier. Controls how much the loss contributes to the final loss
            }

        if loss_bbox is None:
            loss_bbox = { # Regression loss that answers where the object is
                "type": "IoULoss", # How much do the predicted box and GT box overlap
                "loss_weight": 1.0
            }
        
        if loss_centerness is None:
            loss_centerness = { # Centerness loss to find the best center on the object
                "type": "CrossEntropy", # Center points produce better bounding. While edge points produce bad boxes
                "use_sigmoid": True,
                "loss_weight": 1.0
            }
        
        norm_cfg: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,

        if norm_cfg is None:
            norm_cfg = { # This defines how normalization layers in the head are configured
                "type": "GN", # Group Normalization. It divides channels into groups and normalizes them separately. 
                "num_groups": 32, # How many group to divide channels into 
                "requires_grad": True # The scale (γ) and shift (β) parameters in GN are learnable (the network can adjust them).
            }
        
        if init_cfg is None:
            init_cfg = { # This defines how to initialize weights in the head.
                "type"     : "Normal", # Use normal (Gaussian) distribution to initialize weights
                "name"     : "Conv2d", # Applies to all conv layers
                "std"      : 0.01, # Standard deviation of the normal distribution. Small numbers -> start with small weights
                "override" : { # Special rule for classification conv (conv_cls)
                    "type"      : "Normal",
                    "name"      : "conv_cls",
                    "std"       : 0.01,
                    "bias_prob" : 0.01 # Initialize biases so that probability of predicting foreground class is very low initially
                }
            }
