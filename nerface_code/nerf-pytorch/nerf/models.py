import torch


class VeryTinyNeRFModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(VeryTinyNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 65 -> 128)
        self.layer1 = torch.nn.Linear(
            self.xyz_encoding_dims + self.viewdir_encoding_dims, filter_size
        )
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MultiHeadNeRFModel(torch.nn.Module):
    r"""Define a "multi-head" NeRF model (radiance and RGB colors are predicted by
    separate heads).
    """

    def __init__(self, hidden_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(MultiHeadNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims, hidden_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 3_1 (default: 128 -> 1): Predicts radiance ("sigma")
        self.layer3_1 = torch.nn.Linear(hidden_size, 1)
        # Layer 3_2 (default: 128 -> 1): Predicts a feature vector (used for color)
        self.layer3_2 = torch.nn.Linear(hidden_size, hidden_size)

        # Layer 4 (default: 39 + 128 -> 128)
        self.layer4 = torch.nn.Linear(
            self.viewdir_encoding_dims + hidden_size, hidden_size
        )
        # Layer 5 (default: 128 -> 128)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 6 (default: 128 -> 3): Predicts RGB color
        self.layer6 = torch.nn.Linear(hidden_size, 3)

        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x, view = x[..., : self.xyz_encoding_dims], x[..., self.xyz_encoding_dims :]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return torch.cat((x, sigma), dim=-1)


class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        **kwargs
    ):
        super(ReplicateNeRFModel, self).__init__()
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions

        self.dim_xyz = (3 if include_input_xyz else 0) + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if include_input_dir else 0) + 2 * 3 * num_encoding_fn_dir

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, direction = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = self.fc_alpha(x_)
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        rgb = self.fc_rgb(y_)
        return torch.cat((rgb, alpha), dim=-1)


class PaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
    ):
        super(PaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x = xyz#self.relu(self.layers_xyz[0](xyz))
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)




class ConditionalBlendshapePaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True,
        latent_code_dim=32

    ):
        super(ConditionalBlendshapePaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = latent_code_dim

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x,  expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x = xyz#self.relu(self.layers_xyz[0](xyz))
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((xyz, expr_encoding, latent_code), dim=1)
            x = initial
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)




class ConditionalBlendshapePaperSmallerNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """  # Made smaller...

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True,
        latent_code_dim=32

    ):
        super(ConditionalBlendshapePaperSmallerNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = latent_code_dim

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, 256))
        for i in range(1, 5):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir + self.dim_expression, 128))
        for i in range(2):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x,  expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x = xyz#self.relu(self.layers_xyz[0](xyz))
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((xyz, expr_encoding, latent_code), dim=1)
            x = initial
        for i in range(5):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs, expr_encoding), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)












class FlexibleNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
    ):
        super(FlexibleNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu

    def forward(self, x):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ConditionalNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True
    ):
        super(ConditionalNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 1 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        # Encoding for expressions:
        #self.layers_expr = torch.nn.ModuleList()
        #self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        #self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        #self.dim_expression *= 4

        #self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        if self.dim_expression > 0:
            expr_encoding = expr.repeat(xyz.shape[0],1)*(1/3)
            #if self.layers_expr is not None:
            #    for l in self.layers_expr:
            #        expr_encoding = self.relu(l(expr_encoding))
            #if self.layers_expr is not None:
            #    expr_encoding = self.layers_expr[0](expr_encoding)
            #    expr_encoding = self.sigmoid(expr_encoding)
                #expr_encoding = self.layers_expr[1](expr_encoding)
                #expr_encoding = self.sigmoid(expr_encoding)

            #x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)

class ConditionalBlendshapeLearnableCodeNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True,
        latent_code_dim = 32
    ):
        super(ConditionalBlendshapeLearnableCodeNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        #self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        self.dim_latent_code = latent_code_dim
        if not use_viewdirs:
            self.dim_dir = 0

        #Encoding for expressions:
        #self.layers_expr = torch.nn.ModuleList()
        self.layers_expr = None
        #self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        ##self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        #self.dim_expression *= 2

        #self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1): # was num_layers-1
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size + self.dim_expression + self.dim_latent_code, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None, latent_code=None, **kwargs):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        latent_code = latent_code.repeat(xyz.shape[0],1)
        if self.dim_expression > 0:
            expr_encoding = (expr * 1/3).repeat(xyz.shape[0],1)
            #if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            if self.layers_expr is not None:
               expr_encoding = self.layers_expr[0](expr_encoding)
               expr_encoding = torch.nn.functional.tanh(expr_encoding)
               #expr_encoding = self.layers_expr[1](expr_encoding)
               #expr_encoding = self.relu(expr_encoding)

            #x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding, latent_code), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz )):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz,expr_encoding), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ConditionalCompressedBlendshapeLearnableCodeNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True,
        latent_code_dim = 32
    ):
        super(ConditionalCompressedBlendshapeLearnableCodeNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        #self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        self.dim_latent_code = latent_code_dim
        if not use_viewdirs:
            self.dim_dir = 0

        #Encoding for expressions:
        #self.layers_expr = torch.nn.ModuleList()
        self.dim_expression = 10
        self.layer_expr = torch.nn.Linear(76,10)
        #self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        ##self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        #self.dim_expression *= 2

        #self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1): # was num_layers-1
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size + self.dim_expression + self.dim_latent_code, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None, latent_code=None, **kwargs):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        latent_code = latent_code.repeat(xyz.shape[0],1)
        if self.dim_expression > 0:
            expr_encoding = (self.layer_expr(expr)).repeat(xyz.shape[0],1)
            #if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            # if self.layers_expr is not None:
            #    expr_encoding = self.layers_expr[0](expr_encoding)
            #    expr_encoding = torch.nn.functional.tanh(expr_encoding)
            #    #expr_encoding = self.layers_expr[1](expr_encoding)
            #    #expr_encoding = self.relu(expr_encoding)

            #x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding, latent_code), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz )):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz,expr_encoding), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ConditionalCompressedBlendshapeNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True
    ):
        super(ConditionalCompressedBlendshapeNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 20 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        #self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        #Encoding for expressions:
        #self.layers_expr = torch.nn.ModuleList()
        self.dim_expression = 20
        #self.layer_expr = torch.nn.Linear(76,10)

        self.layers_expr = torch.nn.ModuleList()

        self.layers_expr.append(torch.nn.Linear(76, 38))
        self.layers_expr.append(torch.nn.Linear(38, 20))
        self.layers_expr.append(torch.nn.Linear(20, 20))

        #self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        #self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        #self.dim_expression *= 2

        #self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1): # was num_layers-1
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size + self.dim_expression, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None, latent_code=None, **kwargs):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        #latent_code = latent_code.repeat(xyz.shape[0],1)
        if self.dim_expression > 0:
            expr = expr.repeat(xyz.shape[0], 1)

            for expr_layer in self.layers_expr:
                expr = expr_layer(expr)
                expr = self.relu(expr)
            #if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            # if self.layers_expr is not None:
            #    expr_encoding = self.layers_expr[0](expr_encoding)
            #    expr_encoding = torch.nn.functional.tanh(expr_encoding)
            #    #expr_encoding = self.layers_expr[1](expr_encoding)
            #    #expr_encoding = self.relu(expr_encoding)

            #x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            #expr = expr.repeat(xyz.shape[0], 1)
            x = torch.cat((xyz, expr), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz )):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz,expr), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)



class ConditionalBlendshapeNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True
    ):
        super(ConditionalBlendshapeNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        #self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        #Encoding for expressions:
        #self.layers_expr = torch.nn.ModuleList()
        self.layers_expr = None
        #self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        ##self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        #self.dim_expression *= 2

        #self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1): # was num_layers-1
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size + self.dim_expression, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None, **kwargs):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        if self.dim_expression > 0:
            expr_encoding = (expr * 1/3).repeat(xyz.shape[0],1)
            #if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            if self.layers_expr is not None:
               expr_encoding = self.layers_expr[0](expr_encoding)
               expr_encoding = torch.nn.functional.tanh(expr_encoding)
               #expr_encoding = self.layers_expr[1](expr_encoding)
               #expr_encoding = self.relu(expr_encoding)

            #x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz )):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz,expr_encoding), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)














class ConditionalBlendshapeNeRFModel_v2(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True
    ):
        super(ConditionalBlendshapeNeRFModel_v2, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 15 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        #self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        #Encoding for expressions:
        self.layers_expr = torch.nn.ModuleList()
        #self.layers_expr = None
        self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        self.dim_expression *= 4

        #self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers -1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        if self.dim_expression > 0:
            #expr_encoding = (expr * 1/3).repeat(xyz.shape[0],1)
            expr_encoding = (expr * 1/3)#.repeat(xyz.shape[0],1)
            #if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            if self.layers_expr is not None:
               expr_encoding = self.layers_expr[0](expr_encoding)
               expr_encoding = torch.nn.functional.relu(expr_encoding)
               expr_encoding = self.layers_expr[1](expr_encoding)
               expr_encoding = self.relu(expr_encoding)
               expr_encoding = expr_encoding.repeat(xyz.shape[0],1)
            #x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz  )):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.n_down = 5
        # Bx3x256x256 -> Bx128x1x1
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 64

            torch.nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 16

            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 4

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 1

            torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            torch.nn.Tanh()
        )

    def forward(self,x):
        x = self.cnn_layers(x)
        return x

class ConditionalAutoEncoderNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True
    ):
        super(ConditionalAutoEncoderNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 128 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        # Encoding for expressions:
        #self.layers_expr = torch.nn.ModuleList()
        #self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        #self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        #self.dim_expression *= 4

        #self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        if self.dim_expression > 0:
            expr_encoding = expr.repeat(xyz.shape[0],1)
            #if self.layers_expr is not None:
            #    for l in self.layers_expr:
            #        expr_encoding = self.relu(l(expr_encoding))
            #if self.layers_expr is not None:
            #    expr_encoding = self.layers_expr[0](expr_encoding)
            #    expr_encoding = self.sigmoid(expr_encoding)
                #expr_encoding = self.layers_expr[1](expr_encoding)
                #expr_encoding = self.sigmoid(expr_encoding)

            #x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class DiscriminatorModel(torch.nn.Module):
    def __init__(self, dim_latent=32, dim_expressions=76):
        super(DiscriminatorModel, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_latent, dim_latent*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_latent*2, dim_latent*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_latent*2, dim_expressions),
            torch.nn.Tanh(),
        )

    def forward(self, x):

        return self.model(x)
