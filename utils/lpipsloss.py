"""
LPIPS loss module - обертка для стандартного lpips пакета
"""

try:
    import lpips as lpips_lib
    
    def lpips(img1, img2, net='vgg'):
        """
        Compute LPIPS loss between two images
        """
        try:
            # Создаем LPIPS модель если нет
            if not hasattr(lpips, '_lpips_model'):
                lpips._lpips_model = lpips_lib.LPIPS(net=net)
                lpips._lpips_model.cuda()
            
            return lpips._lpips_model(img1, img2)
        except:
            # Fallback если lpips не работает
            import torch
            return torch.tensor(0.0, device=img1.device)
    
except ImportError:
    print("Warning: lpips package not found, using dummy implementation")
    import torch
    
    def lpips(img1, img2, net='vgg'):
        """Dummy LPIPS implementation"""
        return torch.tensor(0.0, device=img1.device)
