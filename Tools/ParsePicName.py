def get_pic_info(pic_name, info_name = 'number'):
    
    info_dic = {
        "number"        : 0,
        "pre_noise"     : 1,
        "material"      : 2,
        "lighting"      : 3,
        "after_noise"   : 4        
    }
    
    temp = pic_name.split(".")[0]
    return temp.split('_')[info_dic.get(info_name)]

# info is a list include 5 elements [number, pre_noise, material, lighting, afternoise]
def generate_pic_info(info):
    

    ans = info[0]+'_'+info[1]+'_'+info[2]+'_'+info[3]+'_'+info[4]+'.jpg'

    
    return ans
    