function [est_X,estR,RRSE] = Fast_LRFMTC(Y,X,maxiters)


randn('state',1); rand('state',1);
dimY = size(Y);
N = ndims(Y);
O = Y;
O(Y~=0)=1;
O_1 = double(tenmat(O,1));
O_2 = double(tenmat(O,2));
O_3 = double(tenmat(O,3));
nObs = sum(O(:));

dscale = sum((Y(:)-sum(Y(:))/nObs).^2)/nObs;
dscale = sqrt(dscale)/N;

Y = Y./dscale;

B = cell(N,1);


Y(O==0) = sum(Y(:))/nObs;
[~,lambda,V] = CP_als(tensor(Y),150);
    B{1} = V{1}*diag(lambda.^(1/3));
    B{2} = V{2}*diag(lambda.^(1/3));
    B{3} = V{3}*diag(lambda.^(1/3));
Y = Y.*O;
    
     est_X = dscale*double(ktensor(B));
     err_X = est_X(:) - X(:);
     rrse0 = sqrt(sum(err_X.^2)/sum(X(:).^2));
     fprintf('rrse0');disp(rrse0); 


num = 0;
S2 = 0;
S3 = 0;

for it = 1:maxiters
    
alpha = 30;       
   
%    %%%%%%%%%%%%%%%%%%%%%%  
    tau1 = 1/norm((B{3}'*B{3}).*(B{2}'*B{2}));
    for l = 1:1
      %  disp(l);
        alpha1(l) = max(alpha*(0.25^(l-1)),1e-8); 
        kr32 = kr(B{3},B{2});
        M{1} = B{1};
        u1   = 1;
        for it1 = 1:15
                B1_past = B{1}; 
                u1_past = u1;
              
                Z1 = M{1} - tau1*(( M{1}*kr32' - double(tenmat(Y,1)) ).*O_1)*kr32;
                          
                [U1,S1,V1] = svd(Z1); 
                                  
                S1 = max((S1 - tau1*alpha1(l)),0);
                B{1} = U1*S1*V1';
                
                u1 = ((1 + 4*(u1^2))^0.5 + 1)/2;
                
                M{1} = B{1} + ((u1_past - 1)/u1)*(B{1} - B1_past);
                
                err_iteration1 = B{1}(:) - B1_past(:);
                LBRelChan1 = sqrt(sum(err_iteration1.^2)/sum(B{1}(:).^2));                            
                %fprintf('LBRelChan1');disp(LBRelChan1);

                if it1 > 5 && LBRelChan1 < 1e-4    
                break;
                end
             
        end

    end
      

     %%%%%%%%%%%%%%%%%%%%%%%    
     
     est_X = dscale*double(ktensor(B));
     err_X = est_X(:) - X(:);
     rrse = sqrt(sum(err_X.^2)/sum(X(:).^2));
     num = num + 1;
     RRSE(num) = rrse;
     fprintf('rrse');disp(rrse); 
     
     
    
     %%%%%%%%%%%%%%%%%%%%%%
      tau2 = 1/norm((B{3}'*B{3}).*(B{1}'*B{1}));
      for l = 1:1
         alpha2(l) = max(alpha*(0.25^(l-1)),1e-8);
         kr31 = kr(B{3},B{1});
         M{2} = B{2};
         u2   = 1;         
         for it2 = 1:15
            %   fprintf('it2');disp(it2);
                B2_past = B{2}; 
                u2_past = u2;
                
               Z2 = M{2} - tau2*(( M{2}*kr31' - double(tenmat(Y,2)) ).*O_2 )*kr31;             
               
               [U2,S2,V2] = svd(Z2);
                           
               S2 = max((S2 - tau2*alpha2(l)),0);
               B{2} = U2*S2*V2';
               
               u2 = ((1 + 4*(u2^2))^0.5 + 1)/2;
                
               M{2} = B{2} + ((u2_past - 1)/u2)*(B{2} - B2_past);              

               err_iteration2 = B{2}(:) - B2_past(:);   
               LBRelChan2 = sqrt(sum(err_iteration2.^2)/sum(B{2}(:).^2));
              % fprintf('LBRelChan2');disp(LBRelChan2);

               if it1 > 5 && LBRelChan2 < 1e-4       
               break;
               end         
         end
      end

    
      %%%%%%%%%%%%%%%%%%%%%%%
     est_X = dscale*double(ktensor(B));
     err_X = est_X(:) - X(:);
     rrse = sqrt(sum(err_X.^2)/sum(X(:).^2));
     num = num + 1;
     RRSE(num) = rrse;     
     fprintf('rrse');disp(rrse);       
     
    %%%%%%%%%%%%%%%%%%%%%%

    tau3 = 1/norm((B{2}'*B{2}).*(B{1}'*B{1}));
    if it == 1
        B{3} = zeros(size(B{3}));
    end
    for l = 1:1
        alpha3(l) = max(alpha*(0.25^(l-1)),1e-8);
        kr21 = kr(B{2},B{1});
        M{3} = B{3};
        u3   = 1;        
        
        for it3 = 1:15
                B3_past = B{3};
                u3_past = u3;
 
                Z3 = M{3} - tau3*(( M{3}*kr21' - double(tenmat(Y,3))).*O_3 )*kr21;

                [U3,S3,V3] = svd(Z3);  
                
                S3 = max((S3 - tau3*alpha3(l)),0);
                B{3} = U3*S3*V3'; 
                
                u3 = ((1 + 4*(u3^2))^0.5 + 1)/2;
                
                M{3} = B{3} + ((u3_past - 1)/u3)*(B{3} - B3_past);                

                err_iteration3 = B{3}(:) - B3_past(:);
                LBRelChan3 = sqrt(sum(err_iteration3.^2)/sum(B{3}(:).^2));
              %  fprintf('LBRelChan3');disp(LBRelChan3);

                if it1 > 5 && LBRelChan3 < 1e-4      
                break;
                end       
        end
    end
 
%     
  %%%%%%%%%%%%%%%%%%%%%%%%
  
  est_X = dscale*double(ktensor(B));
  err_X = est_X(:) - X(:);
  rrse = sqrt(sum(err_X.^2)/sum(X(:).^2));
  num = num + 1;
  RRSE(num) = rrse;  
  fprintf('rrse'); disp(rrse);
  
  
  if it == 1
      normall(:,1) = zeros(prod(dimY),1);
      LBRelChan = 1e10;
      LBRelChan0 = 1e10;
  else         
      normall(:,it) = est_X(:);
      err_iteration = normall(:,it) - normall(:,it-1);
      LBRelChan = sum(err_iteration.^2.^0.5)/sum(abs(normall(:,it))); 
      LBRelChan0 = sqrt(sum(err_iteration.^2)/sum(normall(:,it).^2));
  end 
  fprintf('LBRelChan');disp(LBRelChan);
  fprintf('LBRelChan0');disp(LBRelChan0);
 
  fprintf('Iter. %d: RelChan = %g,R1 = %g,R2 = %g, R3 = %d \n', it, LBRelChan0, sum( diag(S1)/S1(1,1) >  1e-2 ),sum(diag(S2)/S2(1,1) >  1e-2),sum(diag(S3)/S3(1,1) >  1e-2));
 
    estR(1) = sum( diag(S1)/S1(1,1) >  1e-2 );
    estR(2) = sum( diag(S2)/S2(1,1) >  1e-2 );
    estR(3) = sum( diag(S3)/S3(1,1) >  1e-2 );
 
    if it > 5 && LBRelChan0 < 1e-6
           break;
    end
       

end
