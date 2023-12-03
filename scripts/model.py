import torch
import torch.nn as nn
import pytorch_lightning as L


class UserEmbeddings(nn.Module):
    """Vector representation of users."""
    
    def __init__(
        self, 
        user_dim: int, 
        embedding_dim: int,
        dropout: float = 0.2,
        hidden_dim: int = 16,
    ):
        """Initialize the user embeddings.  
        
        Args:
            user_dim (int): Number of features used to represent the user.
            embedding_dim (int): Dimension of the user embedding.
            dropout (float): Dropout rate.
            hidden_dim (int): Dimension of the hidden layer.
        
        """
        super().__init__()
        
        self.seq = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

        
    def forward(self, x) -> torch.Tensor:
        """Get the user embeddings.
        
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, user_dim).
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim).
        """
        return self.seq(x)
    
    
    
class ItemEmbeddings(nn.Module):
    """Vector representation of items (films)."""
    
    def __init__(
        self, 
        item_dim: int, 
        embedding_dim: int,
        dropout: float = 0.2,
        hidden_dim: int = 16,
    ):
        """Initialize the item embeddings.
        
        Args:
            item_dim (int): Number of features used to represent the item.
            embedding_dim (int): Dimension of the item embedding.
            dropout (float): Dropout rate.
            hidden_dim (int): Dimension of the hidden layer.
        """
        super().__init__()
        
        self.seq = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )            

        
    def forward(self, x) -> torch.Tensor:
        """Get the item embeddings.
        
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, item_dim).
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim).
        """
        return self.seq(x)   
        
        
        
class Joint(nn.Module):
    """Joint representation of users and items."""
    
    # Default values for the hidden dimension and dropout rate.
    _DEFAULT_HIDDEN_DIM = 16
    _DEFAULT_DROPOUT = 0.2
    
    @classmethod
    def get_default_hidden_dim(cls):
        return cls._DEFAULT_HIDDEN_DIM
    
    @classmethod
    def get_default_dropout(cls):
        return cls._DEFAULT_DROPOUT 
    
    def __init__(
        self, 
        emb_dim, 
        joint_dim,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        dropout: float = 0.2,
    ):
        """Initialize the layer for the joint representation.
        
        Args:
            emb_dim (int): Dimension of the user and item embeddings.
            joint_dim (int): Dimension of the joint representation.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        self.item = nn.Linear(emb_dim, joint_dim)
        self.user = nn.Linear(emb_dim, joint_dim)
        
        self.ff_head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, joint_dim),
        )
        
    def forward(self, user, item) -> torch.Tensor:
        """Get the joint representation.
        
        Average the item embeddings and concatenate them with the user embeddings
        by summing them. Then, pass the result through a feed-forward head.
        
        Args:
            user (torch.Tensor): Tensor of shape (batch_size, emb_dim).
            item_embeddings (torch.Tensor): Tensor of shape (batch_size, num_items, emb_dim).
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, joint_dim).
        """
        return self.ff_head(self.user(user) + self.item(item))


        
    
class RecModel(nn.Module):
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        embedding_dim: int,
        joint_dim: int,
        dropout: float,
        hidden_dim: int,
    ):
        """Initialize the model.
        
        Args:
            num_user_hidden_layers (int): Number of hidden layers in the user block.
            num_item_hidden_layers (int): Number of hidden layers in the item block.
            user_dim (int): Number of features used to represent the user.
            item_dim (int): Number of features used to represent the item.
            embedding_dim (int): Dimension of the user and item embeddings. 
                Assumes that the user and item embeddings have the same dimension.
            joint_dim (int): Dimension of the joint representation.
            dropout (float): Dropout rate.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super().__init__()
        
        # Multi-layer perceptron for the user embeddings.
        self.user_block = UserEmbeddings(
            user_dim=user_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        
        # Multi-layer perceptron for the item embeddings.
        self.item_block = ItemEmbeddings(
            item_dim=item_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        
        # Layer for the joint representation.
        self.joint = Joint(
            emb_dim=embedding_dim,
            joint_dim=joint_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
    def forward(self, user, item) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the joint representation.
        
        Args:
            user (torch.Tensor): Tensor of shape (batch_size, user_dim).
            items (torch.Tensor): Tensor of shape (batch_size, item_dim).
            
        Returns:
            tuple: Tuple of tensors of shapes:
                - (batch_size, embedding_dim): User embeddings.
                - (batch_size, embedding_dim): Item embeddings.
                - (batch_size, joint_dim): Joint representation.
        """
        user = self.user_block(user)
        item = self.item_block(item)
        return user, item, self.joint(user, item)
    
    
    
class RecRegressionHead(nn.Module):
    """Model with classification head. Predicts whether the user likes the item or not."""
    
    def __init__(
        self, 
        joint_dim: int,
    ):
        """Initialize the model.
        
        Args:
            joint_dim (int): Dimension of the joint representation.
        """
        super().__init__()
        
        # Deep neural network for the classification head.
        self.head = nn.Sequential(
            nn.Linear(joint_dim, joint_dim // 2),
            nn.ReLU(),
            nn.Linear(joint_dim // 2, joint_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(joint_dim // 4),
            nn.Linear(joint_dim // 4, joint_dim // 16),
            nn.ELU(),
            nn.Linear(joint_dim // 16, 1),
        )

    def forward(self, j) -> torch.Tensor:
        """Get the predicted ratings.
        
        Args:
            j (torch.Tensor): Tensor of shape (batch_size, joint_dim).
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1).
        """
        return self.head(j)
    
class RecClassificationHead(nn.Module):
    """Model with classification head. Predicts whether the user likes the item or not."""
    
    def __init__(
        self, 
        joint_dim: int,
    ):
        """Initialize the model.
        
        Args:
            joint_dim (int): Dimension of the joint representation.
        """
        super().__init__()
        
        # Deep neural network for the classification head.
        self.head = nn.Sequential(
            nn.Linear(joint_dim, joint_dim // 2),
            nn.ReLU(),
            nn.Linear(joint_dim // 2, joint_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(joint_dim // 4),
            nn.Linear(joint_dim // 4, joint_dim // 16),
            nn.ELU(),
            nn.Linear(joint_dim // 16, 1),
        )

    def forward(self, j) -> torch.Tensor:
        """Get the predicted ratings.
        
        Args:
            j (torch.Tensor): Tensor of shape (batch_size, joint_dim).
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1).
        """
        return self.head(j)
    
    
class RecModelLightning(L.LightningModule):
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        embedding_dim: int,
        joint_dim: int,
        dropout: float,
        hidden_dim: int,
        lr: float,
        weight_decay: float,
        gamma: float = 1E-4,
    ):
        """Initialize the model.
        
        Args:
            user_dim (int): Number of features used to represent the user.
            item_dim (int): Number of features used to represent the item.
            embedding_dim (int): Dimension of the user and item embeddings. 
                Assumes that the user and item embeddings have the same dimension.
            joint_dim (int): Dimension of the joint representation.
            dropout (float): Dropout rate.
            hidden_dim (int): Dimension of the hidden layers.
            lr (float): Learning rate.
            weight_decay (float): Weight decay.
            gamma (float): Weight of the vector loss.
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = RecModel(
            user_dim=user_dim,
            item_dim=item_dim,
            embedding_dim=embedding_dim,
            joint_dim=joint_dim,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        
        self.output = RecRegressionHead(joint_dim)
        # self.output = RecClassificationHead(joint_dim)
        
        self.output_loss = nn.MSELoss()
        # self.output_loss = nn.BCEWithLogitsLoss()
        self.vec_cos = nn.CosineSimilarity(dim=1)
        
    def forward(
        self,
        user: torch.Tensor,
        items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the predicted ratings.
        
        Args:
            user (torch.Tensor): Tensor of shape (batch_size, user_dim).
            items (torch.Tensor): Tensor of shape (batch_size, item_dim).
            
        Returns:
            tuple: Tuple of tensors of shapes:
                - (batch_size, 1): Predicted ratings.
                - (batch_size, embedding_dim): User embeddings.
                - (batch_size, embedding_dim): Item embeddings.
        """
        
        user, items, j = self.model(user, items)
      
        y = self.output(j)
    
        
        return y, items, user
    
    def training_step(self, batch, batch_idx):
        """Training step.
        
        Args:
            batch (tuple): Tuple of tensors of shapes:
                - (batch_size, user_dim): User features.
                - (batch_size, item_dim): Item features.
                - (batch_size, 1): Ratings.
            batch_idx (int): Batch index.
            
        Returns:
            torch.Tensor: Loss item 
        """
        user, items, labels = batch
        labels = labels.float()
        # Get the predicted ratings.
        y, u, i = self.forward(user, items)
        # Flatten the ratings.
        y = y.flatten()
        
        # Output loss.
        cl_loss = self.output_loss(y, labels)
        # Vector loss.
        vec_loss = 1 - self.vec_cos(u, i).mean()
        # Total loss.
        loss = cl_loss + self.hparams.gamma * vec_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_cl_loss', cl_loss, prog_bar=True)
        self.log('train_vec_loss', vec_loss, prog_bar=True)
        # Classification accuracy.
        self.log('train_acc', (torch.sigmoid(y) >= 0.5).float().eq(labels).float().mean(), prog_bar=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        """Validation step.
        
        Args:
            batch (tuple): Tuple of tensors of shapes:
                - (batch_size, user_dim): User features.
                - (batch_size, item_dim): Item features.
                - (batch_size, 1): Ratings.
            batch_idx (int): Batch index.
            
        Returns:
            torch.Tensor: Loss item
        """
        user, items, labels = batch
        labels = labels.float()
        y, u, i = self.forward(user, items)
        y = y.flatten()
        cl_loss = self.output_loss(y, labels)
        vec_loss = 1 - self.vec_cos(u, i).mean()
        loss = cl_loss# + self.hparams.gamma * vec_loss

        self.log('val_loss', loss)
        self.log('val_cl_loss', cl_loss)
        self.log('val_vec_loss', vec_loss)
        self.log('val_acc', (torch.sigmoid(y) >= 0.5).float().eq(labels).float().mean())
        
        return loss
        
    
    def configure_optimizers(self):
        """Configure the optimizer.
        
        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

        
        
class RecSimpleFF(nn.Module):
    """Simpler model (Two Towers)"""
    
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        hidden_dim: int,
    ):
        """Initialize the model.
        
        Args:
            user_dim (int): Number of features used to represent the user.
            item_dim (int): Number of features used to represent the item.
            hidden_dim (int): Dimension of the hidden layers.
        """
        
        super().__init__()
        # Tower for the user features.
        self.user_to_hidden = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
        )
        
        # Tower for the item features.
        self.item_to_hidden = nn.Sequential(
            nn.Linear(item_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
        )
        
        # Head.
        self.head_mods = []
        x = 2 * hidden_dim
        while x > 1:
            self.head_mods.append(nn.Linear(x, x // 2))
            self.head_mods.append(nn.ELU())
            self.head_mods.append(nn.BatchNorm1d(x // 2))
            self.head_mods.append(nn.Dropout(0.2))
            x = x // 2
            
        self.head_mods.append(nn.Linear(x, 1))
        
        self.head = nn.Sequential(*self.head_mods)
        
    def forward(self, user, item):
        """Get the predicted ratings.
        
        Args:
            user (torch.Tensor): Tensor of shape (batch_size, user_dim).
            items (torch.Tensor): Tensor of shape (batch_size, item_dim).
            
        Returns:
            tuple: Tuple of tensors of shapes:
                - (batch_size, embedding_dim): User embeddings.
                - (batch_size, embedding_dim): Item embeddings.
                - (batch_size, joint_dim): Joint representation.
        """
        user = self.user_to_hidden(user)
        item = self.item_to_hidden(item)
        return user, item, self.head(torch.cat([user, item], dim=1))
    

class RecModelLightningSimpleFF(L.LightningModule):
    """Lightning module for the simpler model."""
    
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        hidden_dim: int,
        lr: float,
        weight_decay: float,
    ):
        """Initialize the model.
        
        Args:
            user_dim (int): Number of features used to represent the user.
            item_dim (int): Number of features used to represent the item.
            hidden_dim (int): Dimension of the hidden layers.
            lr (float): Learning rate.
            weight_decay (float): Weight decay.
        """
        
        super().__init__()
        
        # Save the hyperparameters in self.hparams.
        self.save_hyperparameters()
        
        self.model = RecSimpleFF(
            user_dim=user_dim,
            item_dim=item_dim,
            hidden_dim=hidden_dim,
        )
        
        self.class_loss = nn.MSELoss()
        
    def forward(
        self,
        user: torch.Tensor,
        items: torch.Tensor,
    ) -> torch.Tensor: 
        """Get the predicted ratings."""
        user, items, y = self.model(user, items)
        return y, items, user
    
    def training_step(self, batch, batch_idx):
        user, items, labels = batch
        labels = labels.float()
        y, u, i = self.forward(user, items)
        
        y = y.flatten()
        cl_loss = self.class_loss(y, labels)
        loss = cl_loss

        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        """Validation step.
        
        Args:
            batch (tuple): Tuple of tensors of shapes:
                - (batch_size, user_dim): User features.
                - (batch_size, item_dim): Item features.
                - (batch_size, 1): Ratings.
            batch_idx (int): Batch index.   
            
        Returns:
            torch.Tensor: Loss item
        """
        user, items, labels = batch
        labels = labels.float()
        y, u, i = self.forward(user, items)
        y = y.flatten()
        cl_loss = self.class_loss(y, labels)
        loss = cl_loss

        self.log('val_loss', loss)
        
        return loss
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=8, gamma=0.5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }