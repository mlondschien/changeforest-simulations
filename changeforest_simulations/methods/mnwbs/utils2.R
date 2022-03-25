library(ks)

dist_change_points =  function(Shat,S0)
{
  
  if(length(Shat)==0)
  {return(Inf)}
  
  if(length(S0)==0)
  {return(-Inf)}
  
  temp =rep(0,length(S0))
  for(j in 1:length(S0))
  {
    temp[j] = min(abs(S0[j] - Shat))
  }
  return( max(temp) )
}
#############################################################################

Delta_se_t_1d = function(y,s,e,t,N)
{
  #T =   dim(y)[2]
  n =  dim(y)[2]
  
  n_st = sum(N[s:t])  #n*(t-s+1)
  n_se = sum(N[s:e])  #n*(e-s+1)
  n_te =sum(N[(t+1):e]) #n*(e-(t+1) +1)
  
  aux =  as.vector(y[s:t,])
  aux = aux[which(is.na(aux)==FALSE)]
  temp = ecdf(aux)
  vec_y =  as.vector(y[s:e,])
  vec_y = vec_y[which(is.na(vec_y)==FALSE)]
  Fhat_st =  temp(vec_y)# temp(grid)
  
  aux = y[(t+1):e,]
  aux = aux[which(is.na(aux)==FALSE)]
  temp = ecdf(aux)
  Fhat_te =  temp(vec_y)# temp(grid)
  
  temp =  sqrt( n_st*n_te / n_se   ) *max(abs(Fhat_te - Fhat_st  ))
  
  return(temp )
}


################3

Delta_se_t = function(y,z,s,e,t,h)
{
  
  p =  dim(y)[2]
  # =  dim(y)[]
  
  n_st = (t-s)
  n_se = (e-s)
  n_te = (e-t)
  
  dat  = y[(s+1):t,]
  
  #H.pi <- Hpi(dat)
 # temp = kdensity(dat, start = "gumbel", kernel = "gaussian");
  temp = kde(dat, gridsize = 30, eval.points = z, H = h*diag(p))
  #temp  = mkde(dat[1:100,1:2])
  dat  = y[(t+1):e,]
  temp2 = kde(dat, gridsize = 30, eval.points = z, H = h*diag(p))
  
  val = sqrt((n_st*n_te)/n_se)*max(abs(temp$estimate -temp2$estimate))
  
  return(val)
}

#########################################

wbs_Delta_se_t  =  function(y,z,s,e,t,alpha, beta,h)
{
  alpha_new =  pmax(alpha,s)
  beta_new = pmin(beta,e)
  p =  dim(y)[2]
  # 
  #  ind = which( beta_new- alpha_new > 1+  3*gam)
  #  alpha_new =  alpha_new[ind]
  #  beta_new=   beta_new[ind]
  # # 
  ind = which( beta_new- alpha_new >20)
  alpha_new =  alpha_new[ind]
  beta_new=   beta_new[ind]
  M =  length(alpha_new)
  
  xi = 1/8
  alpha_new2 =alpha_new
  beta_new2  =  beta_new
  alpha_new =   ceiling((1-xi)*alpha_new2+  xi*beta_new2)
  beta_new =    ceiling((1-xi)*beta_new2 +  xi*alpha_new2)
  ind = which( beta_new- alpha_new >1)
  alpha_new =  alpha_new[ind]
  beta_new=   beta_new[ind]
  M =  length(alpha_new)
  
  # print(S)
  
  if(M ==  0)
  {
    return(0) 
  }
  
  b  =  rep(0,M)
  a =  rep(0,M)
  
  #print(beta_new)
  #print(alpha_new)
  #  print()
  
  for( m in 1:M  )
  {
    #  temp  =  rep(0,beta_new[m]-alpha_new[m]+1)
    #  for(t  in    (alpha_new[m]+1):(beta_new[m]-1 )  )
    #  {
    if(alpha_new[m]<t &&  t  < beta_new[m]  )
    {
      b[m] = Delta_se_t(y,z,alpha_new[m],beta_new[m],t,h)
        #Delta_se_t(y,alpha_new[m],beta_new[m],t,N)
    }
    #  }
    # best_ind  =  which.max(temp)
    #  a[m] =  alpha_new[m] +  best_ind
    #  b[m] =   temp[best_ind]
  }
  #best_ind =  which.max(b)
  return(max(b))
  #return(b[best_ind])
}


#######################################3

new_MWBS =  function(y, z,s,e,flag=0,S=NULL,Dval=NULL,pos=1,alpha,beta,h)
{
  #print("pos")
  #print(pos)
  if(s> e)
  {
    S =  c(S,NULL)
    Dval =  c(Dval, NULL)
    pos  = NULL#c(pos,NULL)
    return(list(S=S,pos = pos,Dval = Dval))
  }
  
  alpha_new =  pmax(alpha,s)
  beta_new = pmin(beta,e)
  p =  dim(y)[2]
  # 
  #  ind = which( beta_new- alpha_new > 1+  3*gam)
  #  alpha_new =  alpha_new[ind]
  #  beta_new=   beta_new[ind]
  # # 
  ind = which( beta_new- alpha_new >2*max(p,h^{-p} ))#3*max(p,h^{-p} ) )
  alpha_new =  alpha_new[ind]
  beta_new=   beta_new[ind]
  M =  length(alpha_new)
  
  xi = 1/8
  alpha_new2 =alpha_new
  beta_new2  =  beta_new
  alpha_new =   ceiling((1-xi)*alpha_new2+  xi*beta_new2)
  beta_new =    ceiling((1-xi)*beta_new2 +  xi*alpha_new2)
  ind = which( beta_new- alpha_new >1)
  alpha_new =  alpha_new[ind]
  beta_new=   beta_new[ind]
  M =  length(alpha_new)
  
  # print(S)
  if(M==0 && e-s >2*max(p,h^{-p} ))#3*max(p,h^{-p} )   )
  {
     alpha_new =s
     beta_new =e
  }
  #  beta_new=   beta_new[ind]
  # # 
  #  ind = whic
  if(M ==  0 ||  pos[length(pos)]>7)
  {
    S =  c(S,NULL)
    Dval =  c(Dval, NULL)
    pos  = NULL#c(pos,NULL)
    return(list(S=S,pos = pos,Dval = Dval))
  }
  
  b  =  rep(0,M)
  a =  rep(0,M)
  
  #print(beta_new)
  #print(alpha_new)
  #  print()
  for( m in 1:M  )
  {
    temp  =  rep(0,beta_new[m]-alpha_new[m]+1)
    for(t  in    (alpha_new[m]+   max(h^{-p},p)):(beta_new[m]-max(h^{-p},p) )  )
    {
      temp[t-(alpha_new[m]) ] =  Delta_se_t(y,z,alpha_new[m],beta_new[m],t,h)
        #Delta_se_t(y,alpha_new[m],beta_new[m],t,N)
    }
    best_ind  =  which.max(temp)
    a[m] =  alpha_new[m] +  best_ind
    b[m] =   temp[best_ind]
  }
  best_ind =  which.max(b)
  
  if(abs(a[best_ind]  - s )  <= p   ||  abs(a[best_ind]  - e )  <= p  )
  {
    S =  c(S,NULL)
    Dval =  c(Dval, NULL)
    pos  = NULL#c(pos,NULL)
    return(list(S=S,pos = pos,Dval = Dval))
  }
  #  print(b)
  #  if(b[best_ind]  <tau  )
  # {
  #  return(S)
  #}
  # print(S)
  #best_t =   s+ best_t 
  pos1 =  pos
  pos2 = pos
  pos1[length(pos)] = pos[length(pos)]+1
  pos2[length(pos)] = pos[length(pos)]+1
  
  #print(c(a[best_ind],b[best_ind]))
  temp2 = new_MWBS(y, z,a[best_ind]+p,e,flag,S,Dval,pos2,alpha,beta,h)
    #new_WBS(y,gam,a[best_ind]+1,e,flag, S,Dval,pos2,alpha, beta,N)
  temp1 = new_MWBS(y, z,s,a[best_ind]-p,flag,S,Dval,pos1,alpha,beta,h)
    #new_WBS(y,gam,s,a[best_ind]-1,flag, S,Dval,pos1,alpha, beta,N)
  S1 = temp1$S 
  Dval1 = temp1$Dval     
  pos1 = temp1$pos  
  S2 = temp2$S 
  Dval2 = temp2$Dval 
  pos2 = temp2$pos 
  
  S =  c(S,a[best_ind])
  Dval = c(Dval,b[best_ind])
  
  
  S  =   c(S,S1,S2)
  Dval =  c(Dval,Dval1,Dval2)
  pos = c(pos,pos1,pos2)
  
  return(list(S=S,Dval = Dval,pos=pos))
  # S  =   c(S,S1,S2)
  # 
  # 
  # return(S)
  
}
#####################################################################


new_MBS =  function(y, z,s,e,flag=0,S=NULL,Dval=NULL,pos=1,h)
{
  if(s> e)
  {
    S =  c(S,NULL)
    Dval =  c(Dval, NULL)
    pos  = NULL#c(pos,NULL)
    return(list(S=S,pos = pos,Dval = Dval))
  }
  
  p =  dim(y)[2]
  # 

  #  beta_new=   beta_new[ind]
  # # 
  #  ind = whic
  if( pos[length(pos)]>7)
  {
    S =  c(S,NULL)
    Dval =  c(Dval, NULL)
    pos  = NULL#c(pos,NULL)
    return(list(S=S,pos = pos,Dval = Dval))
  }
  
  M =1
  alpha_new = rep(s,M)
  beta_new = rep(e,M)
  
  b  =  rep(0,M)
  a =  rep(0,M)
  for( m in 1:M  )
  {
    temp  =  rep(0,beta_new[m]-alpha_new[m]+1)
    for(t  in    (alpha_new[m]+   max(h^{-p},p)):(beta_new[m]-max(h^{-p},p) )  )
    {
      temp[t-(alpha_new[m]) ] =  Delta_se_t(y,z,alpha_new[m],beta_new[m],t,h)
      #Delta_se_t(y,alpha_new[m],beta_new[m],t,N)
    }
    best_ind  =  which.max(temp)
    a[m] =  alpha_new[m] +  best_ind
    b[m] =   temp[best_ind]
  }
  best_ind =  which.max(b)
  
  if(abs(a[best_ind]  - s )  <= p   ||  abs(a[best_ind]  - e )  <= p  )
  {
    S =  c(S,NULL)
    Dval =  c(Dval, NULL)
    pos  = NULL#c(pos,NULL)
    return(list(S=S,pos = pos,Dval = Dval))
  }

  pos1 =  pos
  pos2 = pos
  pos1[length(pos)] = pos[length(pos)]+1
  pos2[length(pos)] = pos[length(pos)]+1
  
  #print(c(a[best_ind],b[best_ind]))
  temp2 = new_MBS(y, z,a[best_ind]+p,e,flag,S,Dval,pos2,h)
  #new_WBS(y,gam,a[best_ind]+1,e,flag, S,Dval,pos2,alpha, beta,N)
  temp1 = new_MBS(y, z,s,a[best_ind]-p,flag,S,Dval,pos1,h)
  #new_WBS(y,gam,s,a[best_ind]-1,flag, S,Dval,pos1,alpha, beta,N)
  S1 = temp1$S 
  Dval1 = temp1$Dval     
  pos1 = temp1$pos  
  S2 = temp2$S 
  Dval2 = temp2$Dval 
  pos2 = temp2$pos 
  
  S =  c(S,a[best_ind])
  Dval = c(Dval,b[best_ind])
  
  
  S  =   c(S,S1,S2)
  Dval =  c(Dval,Dval1,Dval2)
  pos = c(pos,pos1,pos2)
  
  return(list(S=S,Dval = Dval,pos=pos))
  # S  =   c(S,S1,S2)
  # 
  # 
  # return(S)
  
}


######################################################################
parent =  function(temp)
{
  p= rep(1,length(temp$pos))
  p[1] = 0
  for(i  in 2:length(temp$pos))
  {
    ind = which(temp$pos[1:(i-1)] == (temp$pos[i]-1))
    p[i] =  ind[length(ind)]
  }
  return(p)
}
########################################################333
thresholding =  function(temp,tau,p)
{
  ind = which(temp$Dval >  tau)
  
  Shat =  c()
  
  if(length(ind)==0)
  {
    return(NULL)
  }
  
  for( j in 1:length(ind))
  {
    if(p[ind[j]]==0)
    {
      Shat = c(Shat,temp$S[ind[j]])
    }
    if(p[ind[j]] > 0  && min(abs(Shat - temp$S[p[ind[j]]] ))==0   )
    {
      Shat = c(Shat,temp$S[ind[j]])
    }
  }
  
  return(Shat)
}

###########################################################3
#new_MWBS(y, z,s,e,flag=0,S=NULL,Dval=NULL,pos=1,alpha,beta,h)
MNWBS_full =  function(y,z,alpha,beta,h,len_tau = 10)
{
    p =  dim(y)[2]
  T = dim(y)[1]
  s = 1
  e = T
  
  
  #grid = 
#  temp1 = new_WBS(z, gam,1,T,0,NULL,NULL,1,alpha,beta,N)
  temp1 = new_MWBS(y, z,s,e,flag=0,S=NULL,Dval=NULL,pos=1,alpha,beta,8*h)
  
  if(length(temp1$S)>1)
  {
    Dval = temp1$Dval
    p1 =  parent(temp1)
    aux = sort(Dval,decreasing = TRUE)
    tau_grid = rev(aux[1:min(len_tau,length(Dval))])- 10^{-30}
    tau_grid =  tau_grid[which(is.na(tau_grid)==FALSE)] 
    #tau_grid  =  tau_grid[which(tau_grid )]
    # tau_grid = c(tau_grid,10)
    
    
    S =  c()
    for( j in 1:length(tau_grid))
    {
      aux = thresholding(temp1,tau_grid[j],p1)
      
      if(length(aux) == 0)
        break;
      
      S[[j]] = sort(aux)
    }
  }
  if(length(temp1$S)==1)
  {
    S = list()
    S[[1]] = sort(temp1$S)
  }
  
  T= dim(y)[1]
  S = unique(S)
  if(length(S)==0)
  {
    return(NULL)
  }  
  best_count = rep(0,length(S)+1)
  
  #for(ind_p in 1:21)
  #{
    indicator = 0
    lamb = 1.8^2
      #log(sum(T))/1.5#2.5#1.5#2#2.555#
   
     h_grid = 1;# 2^seq(-3,3,length = 20)
       #c(h/3,h/2,h/1.5,h,h/1.5,2*h )
 
     #S = rev(S)
   
   
     v_n =  100
     best_count =   matrix(0, length(S)+1,v_n )
     
   #  for(ind_v in 1:v_n  )
    # {
     #  vec = runif(p)
     #  vec =  vec/sqrt(sum(v))
       min_pval = 10
      for(j in 1:length(S))
      {
         if(j == 1 && length(S[[1]])==0 )
         {
           return(B)
         }
         
         B2  =  S[[j]]
         if(j < length(S))
         {
           B1 = S[[j+1]]
         }
         if(j  == length(S))
         {
           B1 = NULL
         }
         temp = setdiff(B2,B1)
         
         st =  10^{-10}
         for(l in 1:length(temp))
         {
           eta =  temp[l]
           
           if( length(B1)==0)
           {
             eta1 = 1
             eta2 = T
           }
           if( length(B1)>0)
           {
             for(k in 1:length(B1))
             {
               if(B1[k]> eta  )
                 break;
             }
             if(B1[k]> eta )
             {
               eta2 = B1[k]
               
               if(k ==1)
                 eta1 = 1
               
               if(k > 1)
                 eta1 = B1[k-1]+1
             }
             if(B1[k]< eta )
             {
               eta1 = B1[k]+1
               eta2 = T
             }######
           }######## if length(B1) >1
           pval  =  rep(0,length(v_n))
           vec_s =  matrix(0,v_n,p)
           for(ind_v in 1:v_n)
           {
             vec = runif(p)
             vec =  vec/sqrt(sum(vec^2))
             vec_s[ind_v,] = vec 
             if(ind_v< p+1)
             {
               vec= rep(0,p)
               vec[ind_v] =1
             }
             #aux = mean(exp(y[eta1:eta,]%*% vec))/ mean(exp(y[(eta+1):eta2,]%*% vec))
             aux = Delta_se_t_1d(y%*%vec,eta1+1,eta2,eta,rep(1,T))  
             #val_aux[ind_v] = aux
             pval[ind_v] = exp(-2*aux^2 )
             
             #exp(all.moments(y[eta1:eta,]%*% vec,5) - all.moments(y[(eta+1):eta2,]%*% vec,5))
               #mean(exp(y[eta1:eta,]%*% vec))/ mean(exp(y[(eta+1):eta2,]%*% vec))
               #all.moments(exp(y[eta1:eta,]%*% vec),5)/all.moments(exp(y[(eta+1):eta2,]%*% vec),5)
             #all.moments(y[eta1:eta,]%*% vec,5)/all.moments(y[(eta+1):eta2,]%*% vec,5)
             #all.moments(exp(y[eta1:eta,]%*% vec),5)/all.moments(exp(y[(eta+1):eta2,]%*% vec),5)
             
             st_aux = aux#max(aux,1/aux )  
             
             # if(st_aux >st)
             # {
             #   st= st_aux
             #   
             # }
             #if(st_aux >1.4)break;
           }## for ind_v
           pval_adj =  p.adjust(pval, method ="fdr" )
           min(pval_adj)
           
           if(min_pval >  min(pval_adj))
           {
             min_pval = min(pval_adj)
           }
         }### for l
         #if( st > sqrt(-0.5* log(0.1/v_n)) )#1.9)
         if(min_pval < 0.0005)#0.001
         {
           break;
         }
#           break;
       }##### j 
       print(min_pval)
     if(min_pval>.1)# (st <1.9)
     {
       return(NULL)
     }
       
     return(S[[j]])
}
#  
# # lamb = log(sum(N))/1.5
#  h2 =  h
#  score =  rep(0,length(S))
#  for(j in 1:(length(S)))
#  {
#    if(length(S[[j]]) == 0 && j ==1)
#    {
#      dat  = y[1:T,]
#      aux = kde(dat, gridsize = 30, eval.points = z, H = h2*diag(p))
#      #aux = kde(dat, gridsize = 30, eval.points = z, H = h*diag(p))
#      score[j] =  sum(log(aux$estimate))
#      j = 2
#    }
#    B  =  S[[j]]
#    
#    k = 1
#    dat  = y[1:B[1],]
#    aux = kde(dat, gridsize = 30, eval.points = z[1:B[1],], H = h2*diag(p))
#    score[j] = score[j]+ sum(log(aux$estimate))
#    
#    if(length(B)>1)
#    {
#        for(k in 1:(length(B)-1))
#        {
#          dat  = y[(B[k]+1):B[k+1],]
#          aux = kde(dat, gridsize = 30, eval.points = z[(B[k]+1):B[k+1],], H = h2*diag(p))
#          score[j] = score[j]+ sum(log(aux$estimate))
#        }
#    }
#    ###
#    dat  = y[(B[k]+1):T,]
#    aux = kde(dat, gridsize = 30, eval.points = z[(B[k]+1):T,], H = h2*diag(p))
#    score[j] = score[j]+ sum(log(aux$estimate))
#    
#  }#
#  best_ind = which.max(score)
#  B= S[[best_ind]]
#  
#  dat  = y[1:T,]
#  aux = kde(dat, gridsize = 30, eval.points = z, H = h2*diag(p))
#  nc_score = sum(log(aux$estimate))
#   
#  if(nc_score > score[best_ind])
#  {
#    B =  NULL
#  }
#  B
#c1 - c2 - Delta_se_t(z,eta1+1,eta2,eta,N)^2 + lamb
MNWBS_full_v2 =  function(y,z,alpha,beta,h,len_tau = 10)
{
  p =  dim(y)[2]
  T = dim(y)[1]
  s = 1
  e = T
  
  
  #grid = 
  #  temp1 = new_WBS(z, gam,1,T,0,NULL,NULL,1,alpha,beta,N)
  temp1 = new_MWBS(y, z,s,e,flag=0,S=NULL,Dval=NULL,pos=1,alpha,beta,8*h)
  
  
  if(length(temp1$S)>1)
  {
    Dval = temp1$Dval
    p1 =  parent(temp1)
    aux = sort(Dval,decreasing = TRUE)
    tau_grid = rev(aux[1:min(len_tau,length(Dval))])- 10^{-30}
    tau_grid =  tau_grid[which(is.na(tau_grid)==FALSE)] 
    #tau_grid  =  tau_grid[which(tau_grid )]
    # tau_grid = c(tau_grid,10)
    
    
    S =  c()
    for( j in 1:length(tau_grid))
    {
      aux = thresholding(temp1,tau_grid[j],p1)
      
      if(length(aux) == 0)
        break;
      
      S[[j]] = sort(aux)
    }
  }
  if(length(temp1$S)==1)
  {
    S = list()
    S[[1]] = sort(temp1$S)
  }
  
  T= dim(y)[1]
  S = unique(S)
  if(length(S)==0)
  {
    return(NULL)
  }  
  best_count = rep(0,length(S)+1)
  
  #for(ind_p in 1:21)
  #{
  indicator = 0
  lamb = 1.8^2
  #log(sum(T))/1.5#2.5#1.5#2#2.555#
 
  
  
  v_n =  400
  best_count =   matrix(0, length(S)+1,v_n )
  
  m_s =  length(S)
  c_point = 0
  for(j in 1:length(S))
  {
    if(j == 1 && length(S[[1]])==0 )
    {
      return(NULL)
    }
    alpha =  min(0.05*m_s/(m_s-j+1),1)
    
    B2  =  S[[j]]
    if(j < length(S))
    {
      B1 = S[[j+1]]
    }
    if(j  == length(S))
    {
      B1 = NULL
    }
    temp = setdiff(B2,B1)
    
    st =  10^{-10}
    pval =  10^10
    for(l in 1:length(temp))
    {
      eta =  temp[l]
      
      if( length(B1)==0)
      {
        eta1 = 1
        eta2 = T
      }
      if( length(B1)>0)
      {
        for(k in 1:length(B1))
        {
          if(B1[k]> eta  )
            break;
        }
        if(B1[k]> eta )
        {
          eta2 = B1[k]
          
          if(k ==1)
            eta1 = 1
          
          if(k > 1)
            eta1 = B1[k-1]+1
        }
        if(B1[k]< eta )
        {
          eta1 = B1[k]+1
          eta2 = T
        }######
      }######## if length(B1) >1
      y_p = y
      z_p = z
      statistics =  rep(0,v_n)
      statistics[1] = -1
      
      if(eta2 -  eta1 >30)
      {
        for(ind_v in 1:v_n)
        {
          y_p = y
          z_p = z
          #vec = runif(p)
          #vec =  vec/sqrt(sum(vec^2))
          indices = sample(  (eta1+15):(eta2-15),eta2-eta1-29 ,replace = FALSE)
          #sample(c((eta1+1):(eta-10),(eta+11):eta2), eta2-eta1-20, replace = FALSE)
          #y_p[c((eta1+1):(eta-10),(eta+11):eta2),] =y[indices,]
          #z_p[c((eta1+1):(eta-10),(eta+11):eta2),] =z[indices,]
          y_p[(eta1+15):(eta2-15),] = y[indices,]
          z_p[(eta1+15):(eta2-15),] = z[indices,]
          #aux = mean(exp(y[eta1:eta,]%*% vec))/ mean(exp(y[(eta+1):eta2,]%*% vec))
          statistics[ind_v] = Delta_se_t(y_p,z_p,eta1+15+1,eta2-15,eta,h/5)
        }## for ind_v
        statistics[1] = Delta_se_t(y,z,eta1+15+1,eta2-15,eta,h/5)
      }
    #  length(which(statistics>statistics[1]))/length(statistics)
      #print(length(which(statistics>statistics[1]))/length(statistics))
      aux_pval = length(which(statistics>statistics[1]))/length(statistics) 
      print(aux_pval)
      if(aux_pval <pval)
      {
        pval  = aux_pval
      }
      #( statistics[1]  < quantile(  statistics,0.12))
          #> quantile(  statistics,0.95) || statistics[1]<  quantile(  statistics,0.05)  )#1.9||| sta )

    }### for l
    if(pval < alpha)
    {
      c_point =1
      break;
    }
    #           break;
  }##### j 
  if(c_point == 1)
  {
    return(S[[j]])
  }

    return(NULL)

}