import torch
from torch import nn

from models import pre_act_resnet, wide_resnet, resnext, densenet

# Used model
#from models import resnet_res5_seblock as resnet
#from models import resnet_res5_seblock_ssa as resnet
#from models import resnet
#from models import resnet_lgdv2 as resnet
#from models import resnet_2p1d_lgdv2 as resnet
from models import resnet_2p1d_lgdv2_cbam as resnet
#from models import resnet_2p1d_lgdv2_cbam_max as resnet
#from models import resnet_2p1d_lgdv2_cbam_reduce as resnet
#from models import resnet_r3d_lgdv2_cbam as resnet
#from models import resnet_2p1d as resnet
#from models import resnet_2p1d_cbam as resnet
#from models import resnet_2p1d_cbam_real as resnet
#from models import slowfastnet as resnet
#from models import p3d_model as p3d

from torch import optim
from torch.optim import lr_scheduler

def generate_model(opt):
    assert opt.model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet', 'slowfast'
    ]
    optimizer = None
    scheduler = None

    #opt.model = 'slowfast'
    #opt.model = 'p3d'
    #from models.slowfastnet import get_fine_tuning_parameters

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet_lgdv2 import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        from models.wide_resnet import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        #from models.resnext import get_fine_tuning_parameters
        from models.resnet_lgdv2 import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        from models.pre_act_resnet import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        from models.densenet import get_fine_tuning_parameters

        if opt.model_depth == 121:
            model = densenet.densenet121(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'slowfast':
        print('use slowfast')
        from models.slowfastnet import get_fine_tuning_parameters
        model = resnet.resnet50(
                class_num=opt.n_classes)

    elif opt.model == 'p3d':
        print('use p3d')
        model = p3d.P3D199(pretrained=False,num_classes=opt.n_classes)

    if not opt.no_cuda:
        model = model.cuda()

        if opt.use_mix:
            from apex import amp
            if opt.nesterov:
                dampening = 0
            else:
                dampening = opt.dampening

            optimizer = optim.SGD(
            model.parameters(),
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
            #tmax = opt.n_epochs//2 + 1
            tmax = opt.n_epochs + 1
            #scheduler = lr_scheduler.ReduceLROnPlateau(
            #optimizer, 'min', patience=opt.lr_patience)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = tmax)
            print("use apex")
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        
        # Old version load pretrain weight(Not r2p1d50 model)
        #model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            #assert opt.arch == pretrain['arch']

            """
            #changed
            model_dict = model.state_dict()
            pretrain_dict = {k:v for k, v in pretrain.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            #model.load_state_dict(checkpoint['state_dict'])
            """
            #model.load_state_dict(pretrain['state_dict'])
            #model.load_state_dict({k: v for k, v in pretrain['state_dict'].items() if v.numel() > 1 and v.shape[0] != 255})

            pre_dict = pretrain['state_dict']
            model_dict = model.state_dict()
            #print(model_dict)
            for name, param in pre_dict.items():
                if name not in model_dict:
                    print('no dict!')
                    print(name)
                    continue
                model_dict[name].copy_(param)

            # New version load pretrain weight(With r2p1d model)
            model = nn.DataParallel(model, device_ids=None)

            if opt.model == 'densenet':
                model.module.classifier = nn.Linear(
                    model.module.classifier.in_features, opt.n_finetune_classes)
                model.module.classifier = model.module.classifier.cuda()
            
            else:
                #model.module.fc = nn.Linear(2048,opt.n_finetune_classes)
                model.module.fc = nn.Linear(2048,opt.n_finetune_classes)
                #model.module.fc = nn.Linear(2304,opt.n_finetune_classes)
                #model.module.fc = nn.Linear(512,opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()
            

            print('hey')
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            if opt.use_mix:
                return model, parameters, optimizer, scheduler
            else:
                return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            #assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if opt.model == 'densenet':
                model.classifier = nn.Linear(
                    model.classifier.in_features, opt.n_finetune_classes)
            '''
            else:
                model.module.fc = nn.Linear(4096,101)
                #model.fc = nn.Linear(4096,101)
                model.module.fc = model.module.fc.cuda()
                #model.fc = model.fc.cuda()
            '''
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    if opt.use_mix:
        return model, model.parameters(), optimizer, scheduler
    else:
        return model, model.parameters()
